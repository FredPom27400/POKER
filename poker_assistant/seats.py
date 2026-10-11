"""Lecture locale (sans IA) des joueurs assis, du bouton et des joueurs en jeu.

Aucune calibration par table : fonctionne quel que soit le nombre de sièges
(3, 5, 6, 8, 9…) et le nombre de joueurs présents.
  * Joueurs assis : chaque joueur a son tapis écrit en gros chiffres jaunes
    (« 115 BB ») juste sous son nom en blanc. On cherche ces paires nom/tapis.
  * Moi : le tapis le plus bas de l'écran.
  * Bouton : le jeton « D » (modèle enregistré par calibrate.py --seats),
    attribué au joueur le plus proche.
  * En jeu : dos de carte (modèle) trouvé au-dessus du nom d'un adversaire.
  * Ordre de jeu : sens horaire autour du centre des joueurs -> ma position.
Les modèles (jeton D, dos de carte) dépendent seulement du thème graphique :
on peut en enregistrer plusieurs (un par thème).
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


def _runs(mask: np.ndarray, w: int, h: int, min_h: float) -> list[tuple[int, int, int, int]]:
    m = cv2.dilate(mask.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 160), 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = [tuple(int(v) for v in st[i][:4]) for i in range(1, n) if st[i][3] >= min_h * h]
    # recolle les morceaux d'une même ligne (« 128 » + « BB »)
    boxes.sort()
    merged: list[list[int]] = []
    for x, y, bw, bh in boxes:
        for m_ in merged:
            mx, my, mw, mh = m_
            same_row = abs((y + bh / 2) - (my + mh / 2)) < 0.4 * max(bh, mh)
            if same_row and x - (mx + mw) < 0.9 * max(bh, mh):
                nx, ny = min(mx, x), min(my, y)
                m_[:] = [nx, ny, max(mx + mw, x + bw) - nx, max(my + mh, y + bh) - ny]
                break
        else:
            merged.append([x, y, bw, bh])
    return [tuple(b) for b in merged if b[2] > b[3] * 1.2]


def find_players(table: np.ndarray) -> list[tuple[float, float, float]]:
    """(x, y, hauteur du texte) de chaque tapis jaune surmonté d'un nom blanc."""
    h, w = table.shape[:2]
    hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
    yellow = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 27) & (hsv[:, :, 1] >= 170) & (hsv[:, :, 2] >= 170)
    white = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 215)
    players = []
    for x, y, bw, bh in _runs(yellow, w, h, 0.02):
        fill = yellow[y:y + bh, x:x + bw].mean()
        if not 0.15 <= fill <= 0.6:          # un texte remplit ~25-45 % de sa boîte (pas un aplat ni un liseré)
            continue
        # le nom occupe la ligne juste au-dessus du tapis, centrée sur lui
        y0, y1 = int(max(0, y - 1.3 * bh)), int(max(0, y - 0.1 * bh))
        x0, x1 = int(max(0, x - 0.2 * bw)), int(min(w, x + 1.2 * bw))
        if y1 - y0 < 3:
            continue
        zone = white[y0:y1, x0:x1]
        cols = zone.any(axis=0)
        if 0.06 <= zone.mean() <= 0.6 and cols.mean() >= 0.3:
            players.append((x + bw / 2, y + bh / 2, float(bh)))
    # un même joueur ne doit apparaître qu'une fois
    uniq: list[tuple[float, float, float]] = []
    for p in sorted(players, key=lambda p: -p[2]):
        if all(abs(p[0] - q[0]) > 2 * q[2] or abs(p[1] - q[1]) > 2 * q[2] for q in uniq):
            uniq.append(p)
    return _drop_inner(uniq)


def _drop_inner(cands: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Les mises sont posées devant les joueurs, vers le centre : tout candidat
    situé à l'intérieur du polygone formé par les autres est une mise."""
    if len(cands) < 4:
        return cands
    pts = np.array([[c[0], c[1]] for c in cands], np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    keep = []
    for c in cands:
        d = cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True)
        if d < c[2]:                       # sur le bord du polygone (à une hauteur de texte près)
            keep.append(c)
    return keep


def _best_match(img: np.ndarray, templates: list[np.ndarray]):
    best = (0.0, (0, 0), None)
    for t in templates:
        if t.shape[0] > img.shape[0] or t.shape[1] > img.shape[1]:
            continue
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (float(score), loc, t)
    return best


class SeatReader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dealers = self._load_all(cfg.get("dealer_templates") or [])
        self.cards = self._load_all(cfg.get("cardback_templates") or [])

    @staticmethod
    def _load_all(paths) -> list[np.ndarray]:
        out = []
        for p in paths:
            if Path(resolve(p)).exists():
                img = cv2.imread(str(resolve(p)))
                if img is not None:
                    out.append(img)
        return out

    @property
    def ready(self) -> bool:
        return bool(self.dealers) and bool(self.cards)

    def read(self, table: np.ndarray) -> SeatFacts | None:
        if not self.ready:
            return None
        players = find_players(table)
        if len(players) < 2:
            return None
        h, w = table.shape[:2]
        xs, ys = [p[0] for p in players], [p[1] for p in players]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        angles = [math.degrees(math.atan2(p[1] - cy, p[0] - cx)) for p in players]
        hero = max(range(len(players)), key=lambda i: players[i][1])
        order = sorted(range(len(players)), key=lambda i: (angles[i] - angles[hero]) % 360)  # sens horaire

        # mise à l'échelle des modèles selon la taille du texte des tapis (fenêtre agrandie/réduite)
        ref = self.cfg.get("ref_text_height") or 0
        med = sorted(p[2] for p in players)[len(players) // 2]
        scale = med / ref if ref else 1.0
        dealers, cards = self.dealers, self.cards
        if abs(scale - 1) > 0.08:
            rs = lambda t: cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # noqa: E731
            dealers, cards = [rs(t) for t in dealers], [rs(t) for t in cards]

        d_score, d_loc, d_tpl = _best_match(table, dealers)
        if d_score < self.cfg.get("dealer_threshold", 0.6):
            return None
        dx, dy = d_loc[0] + d_tpl.shape[1] / 2, d_loc[1] + d_tpl.shape[0] / 2
        owner = min(range(len(players)), key=lambda i: (players[i][0] - dx) ** 2 + (players[i][1] - dy) ** 2)

        in_hand, scores = [], []
        for i in order:
            if i == hero:
                continue
            x, y, th = players[i]
            # les cartes sont au-dessus du nom, lui-même au-dessus du tapis
            x0, x1 = int(max(0, x - 6 * th)), int(min(w, x + 6 * th))
            y0, y1 = int(max(0, y - 6 * th)), int(max(0, y - 0.5 * th))
            score = _best_match(table[y0:y1, x0:x1], cards)[0] if y1 > y0 else 0.0
            scores.append(round(score, 2))
            if score >= self.cfg.get("card_threshold", 0.75):
                in_hand.append(zone_name(angles[i]))

        pos = position_name(len(order), order.index(owner), order.index(hero))
        button = "moi" if owner == hero else zone_name(angles[owner])
        return SeatFacts(pos, button, in_hand, round(d_score, 2), scores)
