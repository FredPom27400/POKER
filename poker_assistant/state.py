"""Logique anti-doublon : un seul appel API par situation de jeu."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .detection import bar_rect


THUMB_WIDTH = 128


def signature(table: np.ndarray, bar_cfg: dict) -> np.ndarray:
    """Empreinte de la table : vignette en niveaux de gris, barre d'actions
    masquée (compte à rebours et montants des boutons ne doivent pas compter)."""
    img = table.copy()
    x, y, w, h = bar_rect(img.shape, bar_cfg)
    img[y:y + h, x:x + w] = 0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = max(1, round(g.shape[0] * THUMB_WIDTH / g.shape[1]))
    small = cv2.resize(g, (THUMB_WIDTH, th), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0)


def change_ratio(a: np.ndarray | None, b: np.ndarray | None, pixel_delta: int = 25) -> float:
    """Part des pixels de la vignette qui ont changé (1.0 si pas de référence)."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(np.count_nonzero(cv2.absdiff(a, b) > pixel_delta)) / a.size


class TurnTracker:
    """Décide quand appeler l'API.

    * nouveau tour (transition non -> oui) : appel, sauf si la table est
      identique à celle du dernier appel (clignotement de la barre) ;
    * pendant un tour : aucun nouvel appel, sauf si recall_on_change est
      activé et que la table a changé puis est restée stable ;
    * après une erreur API : nouvel essai limité, espacé de retry_delay.
    """

    def __init__(self, cfg: dict, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.in_turn = False
        self.last_sig: np.ndarray | None = None    # empreinte du dernier appel
        self.last_call_t = -1e9
        self.pending = False
        self.retries = 0
        self.retry_at: float | None = None
        self._prev_sig: np.ndarray | None = None   # pour vérifier la stabilité

    def _changed(self, a, b) -> bool:
        return change_ratio(a, b, self.cfg["pixel_delta"]) > self.cfg["change_ratio"]

    def should_call(self, is_turn: bool, sig: np.ndarray) -> bool:
        now = self.clock()
        stable = not self._changed(sig, self._prev_sig)
        self._prev_sig = sig

        if not is_turn:
            self.in_turn = False
            self.retry_at = None
            return False

        new_turn = not self.in_turn
        self.in_turn = True
        if self.pending:
            return False
        if now - self.last_call_t < self.cfg["min_seconds_between_calls"]:
            return False

        changed = self._changed(sig, self.last_sig)
        if new_turn and changed:
            return self._fire(sig, now, reset=True)
        if not new_turn and changed and stable and self.cfg.get("recall_on_change"):
            return self._fire(sig, now, reset=True)
        if self.retry_at is not None and now >= self.retry_at:
            return self._fire(sig, now, reset=False)
        return False

    def _fire(self, sig, now, reset: bool) -> bool:
        if reset:
            self.retries = 0
        self.retry_at = None
        self.last_sig = sig
        self.last_call_t = now
        self.pending = True
        return True

    def on_result(self, ok: bool) -> None:
        self.pending = False
        if ok:
            return
        if self.retries < self.cfg["max_retries_per_turn"] and self.in_turn:
            self.retries += 1
            self.retry_at = self.clock() + self.cfg["retry_delay"]
