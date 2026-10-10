"""Génère des tables de poker synthétiques imitant le client (pour les tests).

Tour : barre avec 3 boutons rouges FOLD / CALL / RAISE + compte à rebours.
Attente : barre « Présélection de la prochaine action » avec cases à cocher.
Des éléments rouges (cartes coeur/carreau, jetons) sont ajoutés hors de la barre
comme pièges pour le détecteur.
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 1280, 720
BAR = {"x": 0.45, "y": 0.88, "width": 0.55, "height": 0.12}


def _card(img, x, y, label, red):
    cv2.rectangle(img, (x, y), (x + 50, y + 70), (250, 250, 250), -1)
    cv2.putText(img, label, (x + 6, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (30, 30, 200) if red else (20, 20, 20), 2)


def make_table(turn: bool, seed: int = 0, board: int = 3, timer: int = 15,
               w: int = W, h: int = H, ring: float | None = None,
               seat_clock: int | None = None, gap: int | None = None) -> np.ndarray:
    """ring : fraction restante du chrono circulaire autour de mon avatar ;
    seat_clock : secondes affichées à côté de mon avatar (hors barre d'actions)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (35, 30, 25), np.uint8)
    cv2.ellipse(img, (w // 2, int(h * 0.45)), (int(w * 0.42), int(h * 0.33)), 0, 0, 360, (40, 110, 30), -1)
    for i in range(board):
        _card(img, int(w * 0.36) + i * 60, int(h * 0.38), "AKQJT98"[(i + seed) % 7], red=(i + seed) % 2 == 0)
    _card(img, int(w * 0.46), int(h * 0.66), "A", True)
    _card(img, int(w * 0.46) + 55, int(h * 0.66), "K", True)
    for _ in range(6):  # jetons rouges sur le tapis
        c = (int(rng.integers(int(w * .2), int(w * .8))), int(rng.integers(int(h * .2), int(h * .7))))
        cv2.circle(img, c, 12, (40, 40, 210), -1)
    cv2.putText(img, f"Pot: {int(rng.integers(3, 80))} BB", (int(w * 0.44), int(h * 0.33)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    seat = (int(w * 0.3), int(h * 0.78))  # mon avatar
    cv2.circle(img, seat, 32, (90, 90, 90), -1)
    if ring is not None and ring > 0:
        cv2.ellipse(img, seat, (42, 42), -90, 0, 360 * ring, (0, 200, 255), 8)
    if seat_clock is not None:
        cv2.putText(img, f"{seat_clock:02d}", (seat[0] + 55, seat[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    bx, by = int(BAR["x"] * w), int(BAR["y"] * h)
    bw, bh = int(BAR["width"] * w), int(BAR["height"] * h)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (45, 40, 38), -1)
    if turn:
        gap = bw // 40 if gap is None else gap   # gap=4 : boutons presque collés (table Expresso)
        btn_w = (bw - 4 * gap) // 3
        for i, label in enumerate(["FOLD", "CALL 2", "RAISE TO 6"]):
            x0 = bx + gap + i * (btn_w + gap)
            cv2.rectangle(img, (x0, by + gap), (x0 + btn_w, by + bh - gap), (35, 35, 200), -1)
            cv2.putText(img, label, (x0 + 10, by + bh // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"{timer}s", (bx - 50, by + bh // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(img, "Preselection de la prochaine action", (bx + 15, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        for i, label in enumerate(["Check/Fold", "Check", "Call any"]):
            x0 = bx + 15 + i * (bw // 3)
            cv2.rectangle(img, (x0, by + 45), (x0 + 16, by + 61), (200, 200, 200), 2)
            cv2.putText(img, label, (x0 + 24, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def make_screen(table: np.ndarray, x: int = 200, y: int = 150, sw: int = 1920, sh: int = 1080) -> np.ndarray:
    screen = np.full((sh, sw, 3), (90, 90, 90), np.uint8)
    screen[y:y + table.shape[0], x:x + table.shape[1]] = table
    return screen
