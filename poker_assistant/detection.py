"""Détection locale « c'est mon tour » sur la barre d'actions.

Deux signaux, combinables :
  * couleur : quand c'est mon tour la barre contient plusieurs gros boutons
    rouges (FOLD / CHECK-CALL / BET-RAISE). En mode présélection elle n'affiche
    que du texte et des cases à cocher -> presque pas de rouge.
  * template : corrélation (TM_CCOEFF_NORMED) avec la barre de référence
    enregistrée par calibrate.py (turn_bar.png, et wait_bar.png si présent).

Le mode « hybrid » (défaut) exige le signal couleur et utilise le template,
s'il existe, comme garde-fou contre les faux positifs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import resolve


@dataclass
class Detection:
    is_turn: bool
    red_ratio: float = 0.0
    buttons: int = 0
    turn_score: float | None = None
    wait_score: float | None = None
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    bar_width: int = 1

    @property
    def layout(self) -> tuple[int, ...]:
        """Disposition des boutons rouges : centres horizontaux arrondis à 1/20
        de la barre. Change quand on passe p.ex. de CHECK/BET à FOLD/CALL/RAISE."""
        return tuple(sorted(round(20 * (x + w / 2) / self.bar_width) for x, _, w, _ in self.boxes))

    def summary(self) -> str:
        s = f"tour={'OUI' if self.is_turn else 'non'} rouge={self.red_ratio:.3f} boutons={self.buttons}"
        if self.turn_score is not None:
            s += f" tpl_tour={self.turn_score:.2f}"
        if self.wait_score is not None:
            s += f" tpl_attente={self.wait_score:.2f}"
        return s


def bar_rect(table_shape: tuple[int, ...], bar: dict) -> tuple[int, int, int, int]:
    """Convertit la barre (fractions) en pixels (x, y, w, h) dans l'image de table."""
    h, w = table_shape[:2]
    x = int(round(bar["x"] * w))
    y = int(round(bar["y"] * h))
    bw = max(1, int(round(bar["width"] * w)))
    bh = max(1, int(round(bar["height"] * h)))
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    return x, y, min(bw, w - x), min(bh, h - y)


def crop_bar(table: np.ndarray, bar: dict) -> np.ndarray:
    x, y, w, h = bar_rect(table.shape, bar)
    return table[y:y + h, x:x + w]


def red_mask(bgr: np.ndarray, det: dict) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hue_ok = (h <= det["red_hue_low"]) | (h >= det["red_hue_high"])
    mask = (hue_ok & (s >= det["red_sat_min"]) & (v >= det["red_val_min"])).astype(np.uint8) * 255
    # Le texte blanc des boutons troue le masque : fermeture pour recoller.
    k = max(3, (min(bgr.shape[:2]) // 12) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return mask


def red_buttons(bgr: np.ndarray, det: dict) -> tuple[float, list[tuple[int, int, int, int]]]:
    mask = red_mask(bgr, det)
    area = mask.shape[0] * mask.shape[1]
    ratio = float(np.count_nonzero(mask)) / area
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= det["min_button_area"] * area and w >= h * 0.8:  # boutons plutôt larges
            boxes.append((int(x), int(y), int(w), int(h)))
    return ratio, boxes


def _gray_norm(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA)


def template_score(bar: np.ndarray, template: np.ndarray) -> float:
    """Corrélation normalisée entre la barre et un template (redimensionné)."""
    size = (bar.shape[1], bar.shape[0])
    a = _gray_norm(bar, size).astype(np.float32)
    b = _gray_norm(template, size).astype(np.float32)
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


class TurnDetector:
    def __init__(self, det_cfg: dict, bar_cfg: dict):
        self.cfg = det_cfg
        self.bar = bar_cfg
        self.turn_tpl = self._load(det_cfg.get("template_turn"))
        self.wait_tpl = self._load(det_cfg.get("template_wait"))

    @staticmethod
    def _load(path: str | None) -> np.ndarray | None:
        if not path:
            return None
        p = resolve(path)
        return cv2.imread(str(p)) if Path(p).exists() else None

    def detect(self, table: np.ndarray) -> Detection:
        bar = crop_bar(table, self.bar)
        ratio, boxes = red_buttons(bar, self.cfg)
        color_ok = ratio >= self.cfg["red_ratio_min"] and len(boxes) >= self.cfg["min_buttons"]

        ts = template_score(bar, self.turn_tpl) if self.turn_tpl is not None else None
        ws = template_score(bar, self.wait_tpl) if self.wait_tpl is not None else None
        tpl_ok: bool | None = None
        if ts is not None:
            tpl_ok = ts >= self.cfg["template_min_score"] and (ws is None or ts > ws)
        elif ws is not None:
            tpl_ok = ws < self.cfg["template_min_score"]

        method = self.cfg.get("method", "hybrid")
        if method == "color" or tpl_ok is None:
            is_turn = color_ok
        elif method == "template":
            is_turn = tpl_ok
        else:  # hybrid
            is_turn = color_ok and tpl_ok

        return Detection(is_turn, ratio, len(boxes), ts, ws, boxes, bar.shape[1])


class Debouncer:
    """Ne valide un changement d'état qu'après N frames identiques consécutives."""

    def __init__(self, frames: int):
        self.frames = max(1, int(frames))
        self.state = False
        self._candidate = False
        self._count = 0

    def update(self, raw: bool) -> bool:
        if raw == self.state:
            self._count = 0
        else:
            if raw != self._candidate:
                self._candidate, self._count = raw, 0
            self._count += 1
            if self._count >= self.frames:
                self.state, self._count = raw, 0
        return self.state

    def force(self, state: bool) -> None:
        self.state, self._candidate, self._count = state, state, 0


def annotate(table: np.ndarray, bar_cfg: dict, d: Detection) -> np.ndarray:
    """Image de debug : rectangle de la barre + boutons détectés + scores."""
    out = table.copy()
    x, y, w, h = bar_rect(table.shape, bar_cfg)
    color = (0, 255, 0) if d.is_turn else (0, 165, 255)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    for bx, by, bw, bh in d.boxes:
        cv2.rectangle(out, (x + bx, y + by), (x + bx + bw, y + by + bh), (255, 255, 0), 2)
    cv2.putText(out, d.summary(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, d.summary(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return out
