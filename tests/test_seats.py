import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_assistant.seats import SeatReader, position_name, zone_name  # noqa: E402

# 5 sièges comme sur la table de l'utilisateur : moi en bas, puis sens horaire.
W, H = 1600, 1000
SEATS = [(800, 740), (300, 560), (500, 110), (1150, 110), (1350, 560)]


@pytest.mark.parametrize("n,btn,expected", [
    (5, 0, "BTN"), (5, 4, "SB"), (5, 3, "BB"), (5, 2, "UTG"), (5, 1, "CO"),
    (6, 3, "UTG"), (6, 2, "HJ"), (6, 1, "CO"), (2, 0, "BTN"), (2, 1, "BB"), (4, 1, "CO"),
])
def test_position_name(n, btn, expected):
    # hero = index 0 dans l'ordre du jeu ; btn = index du bouton
    assert position_name(n, btn, 0) == expected


def test_zone_name():
    assert zone_name(90) == "en bas" and zone_name(180) == "à gauche"
    assert zone_name(-135) == "en haut à gauche" and zone_name(-45) == "en haut à droite"
    assert zone_name(0) == "à droite"


def _templates(tmp_path):
    rng = np.random.default_rng(0)
    # Motifs lisses (comme de vraies icônes) pour rester reconnaissables une fois redimensionnés.
    dealer = cv2.GaussianBlur(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8), (9, 9), 0)
    card = cv2.GaussianBlur(rng.integers(0, 255, (44, 66, 3), dtype=np.uint8), (9, 9), 0)
    dealer = cv2.normalize(dealer, None, 0, 255, cv2.NORM_MINMAX)
    card = cv2.normalize(card, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(tmp_path / "d.png"), dealer)
    cv2.imwrite(str(tmp_path / "c.png"), card)
    return dealer, card


def _table(dealer, card, btn_seat, with_cards, scale=1.0):
    img = np.full((H, W, 3), (60, 40, 20), np.uint8)
    bx, by = SEATS[btn_seat]
    cx, cy = int(bx + (800 - bx) * 0.25), int(by + (450 - by) * 0.25)   # jeton devant le joueur
    img[cy - 20:cy + 20, cx - 20:cx + 20] = dealer
    for i in with_cards:
        x, y = SEATS[i]
        img[y - 70:y - 26, x - 33:x + 33] = card       # dos de carte au-dessus du nom
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _reader(tmp_path):
    return SeatReader({"points": [(x / W, y / H) for x, y in SEATS], "ref_width": W,
                       "dealer_template": str(tmp_path / "d.png"), "cardback_template": str(tmp_path / "c.png"),
                       "dealer_threshold": 0.6, "card_threshold": 0.75})


@pytest.mark.parametrize("btn,cards,pos,opp", [
    (0, [1, 2], "BTN", 2), (2, [1, 2, 3, 4], "UTG", 4), (1, [1, 3], "CO", 2), (3, [3], "BB", 1),
])
def test_seat_reader(tmp_path, btn, cards, pos, opp):
    dealer, card = _templates(tmp_path)
    facts = _reader(tmp_path).read(_table(dealer, card, btn, cards))
    assert (facts.position, facts.opponents) == (pos, opp)
    assert "ma position = " + pos in facts.prompt()


def test_seat_reader_other_window_size(tmp_path):
    dealer, card = _templates(tmp_path)
    facts = _reader(tmp_path).read(_table(dealer, card, 4, [1, 4], scale=0.8))
    assert (facts.position, facts.opponents) == ("SB", 2)


def test_not_ready_without_templates(tmp_path):
    r = SeatReader({"points": [(0.5, 0.9), (0.2, 0.5)], "dealer_template": str(tmp_path / "x.png"),
                    "cardback_template": str(tmp_path / "y.png")})
    assert not r.ready and r.read(np.zeros((10, 10, 3), np.uint8)) is None
