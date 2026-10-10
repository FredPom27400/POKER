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

    * nouveau tour (la barre a disparu plus de blink_frames captures) : appel
      si la table a changé depuis le dernier appel ;
    * tour « continu » (barre restée affichée, ou disparue un instant parce
      que les adversaires ont joué en pré-action) : nouvel appel seulement
      après un changement BRUSQUE de la table (jetons, carte, fold), resté
      stable recall_stable_frames captures, ou si les boutons ont changé.
      Les timers ne déclenchent rien : un timer qui se vide ne change que
      quelques pixels par capture (pas brusque), et des chiffres qui défilent
      changent toujours les mêmes pixels, marqués « volatils » et ignorés.
      Un changement survenu pendant l'appel API est absorbé (tu n'as pas
      encore pu jouer) ;
    * après une erreur API : nouvel essai limité, espacé de retry_delay.
    """

    def __init__(self, cfg: dict, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.in_turn = False
        self.last_sig: np.ndarray | None = None    # empreinte du dernier appel
        self.last_layout: tuple = ()
        self.last_call_t = -1e9
        self.pending = False
        self.retries = 0
        self.retry_at: float | None = None
        self.calls_this_turn = 0
        self.reason = ""
        self.off_frames = 1 << 30
        self._prev: tuple | None = None            # (sig, layout) de la capture précédente
        self._stable = 0
        self._counts: np.ndarray | None = None     # nb de changements par pixel
        self._jump: np.ndarray | None = None       # pixels touchés par un changement brusque
        self._absorb = False                       # changement survenu pendant l'appel API

    def _changed(self, a, b) -> bool:
        return change_ratio(a, b, self.cfg["pixel_delta"]) > self.cfg["change_ratio"]

    def _layout_changed(self, layout: tuple) -> bool:
        if not layout or layout == self.last_layout:
            return False
        return not set(layout) <= set(self.last_layout)   # bouton manquant = survol

    def _reset_motion(self, sig: np.ndarray) -> None:
        self._counts = np.zeros(sig.shape, np.uint8)
        self._jump = np.zeros(sig.shape, bool)

    def _observe(self, sig: np.ndarray, layout: tuple) -> None:
        if self._prev is None or self._prev[0].shape != sig.shape:
            self._reset_motion(sig)
            self._stable = 0
            self._prev = (sig, layout)
            return
        diff = cv2.absdiff(sig, self._prev[0]) > self.cfg["pixel_delta"]
        self._counts = np.minimum(self._counts.astype(np.uint16) + diff, 255).astype(np.uint8)
        step = diff & (self._counts < 2)                   # hors pixels volatils
        if step.mean() > self.cfg["change_ratio"]:
            self._jump |= step
            self._stable = 0
        elif layout == self._prev[1]:
            self._stable += 1
        else:
            self._stable = 0
        self._prev = (sig, layout)

    def _jump_confirmed(self, sig: np.ndarray) -> bool:
        if self._jump is None or not self._jump.any():
            return False
        real = self._jump & (self._counts < 2)
        if self.last_sig is not None and self.last_sig.shape == sig.shape:
            real &= cv2.absdiff(sig, self.last_sig) > self.cfg["pixel_delta"]
        if real.mean() > self.cfg["change_ratio"]:
            return True
        self._jump[:] = False
        return False

    def should_call(self, is_turn: bool, sig: np.ndarray, layout: tuple = ()) -> bool:
        now = self.clock()
        self._observe(sig, layout)

        if not is_turn:
            self.in_turn = False
            self.off_frames += 1
            self.retry_at = None
            return False

        new_turn = not self.in_turn and self.off_frames > self.cfg["blink_frames"]
        self.in_turn = True
        self.off_frames = 0
        if new_turn:
            self.calls_this_turn = 0
            self._reset_motion(sig)
        if self.pending:
            return False
        if now - self.last_call_t < self.cfg["min_seconds_between_calls"]:
            return False

        if new_turn:
            if self._changed(sig, self.last_sig):
                return self._fire(sig, layout, now, "nouveau tour")
            self.last_layout = layout or self.last_layout
            return False
        if self.retry_at is not None and now >= self.retry_at:
            return self._fire(sig, layout, now, "nouvel essai", reset=False)
        if self.calls_this_turn >= self.cfg["max_calls_per_turn"]:
            return False
        if self._stable < self.cfg["recall_stable_frames"]:
            return False
        if self.cfg.get("recall_on_layout") and self._layout_changed(layout):
            return self._fire(sig, layout, now, "boutons modifiés")
        if self._absorb:
            # La table a bougé pendant que Claude répondait (animation, étiquette
            # d'action qui s'efface…) : tu n'as pas encore pu jouer, ce n'est donc
            # pas une nouvelle décision. On prend l'état stabilisé comme référence.
            self._absorb = False
            self.last_sig = sig
            self._jump[:] = False
            return False
        if self.cfg.get("recall_on_table_change") and self._jump_confirmed(sig):
            return self._fire(sig, layout, now, "table modifiée sans fin de tour visible")
        return False

    def _fire(self, sig, layout, now, reason: str, reset: bool = True) -> bool:
        if reset:
            self.retries = 0
        self.reason = reason
        self.retry_at = None
        self.last_sig = sig
        self.last_layout = layout
        self.last_call_t = now
        self.calls_this_turn += 1
        self.pending = True
        if self._jump is not None:
            self._jump[:] = False
        return True

    def on_result(self, ok: bool) -> None:
        self.pending = False
        self._absorb = self._jump is not None and bool(self._jump.any())
        if ok:
            return
        if self.retries < self.cfg["max_retries_per_turn"] and self.in_turn:
            self.retries += 1
            self.retry_at = self.clock() + self.cfg["retry_delay"]
