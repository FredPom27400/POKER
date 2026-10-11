"""Lecture locale (sans IA) du bouton et des joueurs encore en jeu.

Calibré une fois par calibrate.py --seats :
  * la position de chaque siège (clic sur le nom de chaque joueur, moi compris) ;
  * un modèle du jeton « D » et un modèle du dos de carte d'un adversaire.

À chaque appel :
  * le jeton D est cherché sur toute la table (template matching) ; son
    propriétaire est le siège le plus proche ;
  * un adversaire est « en jeu » si un dos de carte est trouvé juste au-dessus
    de son nom ;
  * l'ordre de jeu est le sens horaire autour du centre de la table, ce qui
    donne ma position (BTN, SB, BB, UTG, HJ, CO…).
Ces faits sont transmis à Claude comme informations fiables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import resolve

ZONES = ["à droite", "en bas à droite", "en bas", "en bas à gauche",
         "à gauche", "en haut à gauche", "en haut", "en haut à droite"]   # par pas de 45° (y vers le bas)


@dataclass
class SeatFacts:
    position: str                 # ma position (BTN, SB, BB, UTG, HJ, CO, MP…)
    button: str                   # emplacement du bouton (« à gauche », « moi »…)
    in_hand: list[str]            # emplacements des adversaires encore en jeu
    dealer_score: float
    card_scores: list[float]

    @property
    def opponents(self) -> int:
        return len(self.in_hand)

    def prompt(self) -> str:
        where = ", ".join(self.in_hand) if self.in_hand else "aucun"
        return (f"Faits mesurés automatiquement sur l'image (fiables, prioritaires sur ta propre lecture) : "
                f"ma position = {self.position} ; bouton : {self.button} ; "
                f"adversaires encore en jeu = {self.opponents} ({where}).")


def zone_name(angle_deg: float) -> str:
    return ZONES[round((angle_deg % 360) / 45) % 8]


def position_name(n: int, btn: int, hero: int) -> str:
    """Position du siège `hero` parmi n joueurs, sièges indexés dans le sens du jeu."""
    if n == 2:
        return "BTN" if hero == btn else "BB"   # en tête-à-tête, le bouton est petite blinde
    k = (hero - btn) % n                          # 0 = BTN, 1 = SB, 2 = BB, 3 = UTG…
    if k == 0:
        return "BTN"
    if k == 1:
        return "SB"
    if k == 2:
        return "BB"
    before_btn = n - k                            # 1 = CO, 2 = HJ, 3 = MP…
    if k == 3:
        return "UTG" if n >= 5 else "CO"         # premier à parler (à 4, c'est aussi le CO)
    return {1: "CO", 2: "HJ"}.get(before_btn, "MP")


class SeatReader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.points = [tuple(p) for p in cfg.get("points") or []]
        self.dealer = self._load(cfg.get("dealer_template"))
        self.cardback = self._load(cfg.get("cardback_template"))

    @staticmethod
    def _load(path):
        if not path or not Path(resolve(path)).exists():
            return None
        return cv2.imread(str(resolve(path)))

    @property
    def ready(self) -> bool:
        return len(self.points) >= 2 and self.dealer is not None and self.cardback is not None

    def _scaled(self, tpl: np.ndarray, table: np.ndarray) -> np.ndarray:
        ref_w = self.cfg.get("ref_width") or table.shape[1]
        s = table.shape[1] / ref_w
        if abs(s - 1) < 0.02:
            return tpl
        return cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def read(self, table: np.ndarray) -> SeatFacts | None:
        if not self.ready:
            return None
        h, w = table.shape[:2]
        pts = [(x * w, y * h) for x, y in self.points]
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        angles = [math.degrees(math.atan2(y - cy, x - cx)) for x, y in pts]
        hero = max(range(len(pts)), key=lambda i: pts[i][1])          # siège le plus bas = moi
        order = sorted(range(len(pts)), key=lambda i: (angles[i] - angles[hero]) % 360)  # sens horaire

        dealer = self._scaled(self.dealer, table)
        res = cv2.matchTemplate(table, dealer, cv2.TM_CCOEFF_NORMED)
        _, d_score, _, d_loc = cv2.minMaxLoc(res)
        if d_score < self.cfg.get("dealer_threshold", 0.6):
            return None
        dx, dy = d_loc[0] + dealer.shape[1] / 2, d_loc[1] + dealer.shape[0] / 2
        owner = min(range(len(pts)), key=lambda i: (pts[i][0] - dx) ** 2 + (pts[i][1] - dy) ** 2)

        card = self._scaled(self.cardback, table)
        th, tw = card.shape[:2]
        in_hand, scores = [], []
        for i in order:
            if i == hero:
                continue
            x, y = pts[i]
            x0, x1 = int(max(0, x - 2 * tw)), int(min(w, x + 2 * tw))
            y0, y1 = int(max(0, y - 4 * th)), int(min(h, y + 0.5 * th))
            roi = table[y0:y1, x0:x1]
            score = 0.0
            if roi.shape[0] >= th and roi.shape[1] >= tw:
                score = float(cv2.minMaxLoc(cv2.matchTemplate(roi, card, cv2.TM_CCOEFF_NORMED))[1])
            scores.append(round(score, 2))
            if score >= self.cfg.get("card_threshold", 0.6):
                in_hand.append(zone_name(angles[i]))

        n = len(order)
        pos = position_name(n, order.index(owner), order.index(hero))
        button = "moi" if owner == hero else zone_name(angles[owner])
        return SeatFacts(pos, button, in_hand, round(d_score, 2), scores)
