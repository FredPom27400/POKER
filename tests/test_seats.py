import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_assistant.seats import SeatReader, find_players, position_name, zone_name  # noqa: E402

W, H = 1600, 1000
CENTER = (800, 440)
YELLOW = (40, 200, 255)    # BGR du texte des tapis (teinte ~22, comme sur le client)


@pytest.mark.parametrize("n,btn,expected", [
    (5, 0, "BTN"), (5, 4, "SB"), (5, 3, "BB"), (5, 2, "UTG"), (5, 1, "CO"),
    (6, 3, "UTG"), (6, 2, "HJ"), (6, 1, "CO"), (2, 0, "BTN"), (2, 1, "BB"), (4, 1, "CO"), (3, 2, "SB"),
])
def test_position_name(n, btn, expected):
    assert position_name(n, btn, 0) == expected   # moi = index 0 dans l'ordre du jeu


def test_zone_name():
    assert zone_name(90) == "en bas" and zone_name(180) == "à gauche"
    assert zone_name(-135) == "en haut à gauche" and zone_name(-45) == "en haut à droite"


def seat_xy(i: int, n: int) -> tuple[int, int]:
    """Siège i sur n autour d'une table ovale ; i = 0 en bas (moi), sens horaire à l'écran."""
    a = math.radians(90 + 360 * i / n)
    return int(CENTER[0] + 620 * math.cos(a)), int(CENTER[1] + 330 * math.sin(a))


def _templates(tmp_path):
    rng = np.random.default_rng(0)
    smooth = lambda shape: cv2.normalize(cv2.GaussianBlur(  # noqa: E731
        rng.integers(0, 255, shape, dtype=np.uint8), (9, 9), 0), None, 0, 255, cv2.NORM_MINMAX)
    dealer, card = smooth((40, 40, 3)), smooth((44, 66, 3))
    cv2.imwrite(str(tmp_path / "d.png"), dealer)
    cv2.imwrite(str(tmp_path / "c.png"), card)
    return dealer, card


def make_table(dealer, card, n_seats, occupied, btn, with_cards, bets=()):
    img = np.full((H, W, 3), (60, 40, 20), np.uint8)
    cv2.ellipse(img, CENTER, (560, 280), 0, 0, 360, (120, 70, 30), -1)
    for i in occupied:
        x, y = seat_xy(i, n_seats)
        cv2.putText(img, "Joueur%d" % i, (x - 70, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "%d BB" % (100 + i), (x - 60, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, YELLOW, 3)
        if i in with_cards:
            img[y - 85:y - 41, x - 33:x + 33] = card
    for i in bets:   # mise devant le joueur, avec un jeton blanc au-dessus du montant
        x, y = seat_xy(i, n_seats)
        bx, by = int(x + (CENTER[0] - x) * 0.35), int(y + (CENTER[1] - y) * 0.35)
        cv2.circle(img, (bx, by - 30), 14, (255, 255, 255), -1)
        cv2.putText(img, "2 BB", (bx - 35, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
    x, y = seat_xy(btn, n_seats)
    dx, dy = int(x + (CENTER[0] - x) * 0.2), int(y + (CENTER[1] - y) * 0.2)
    img[dy - 20:dy + 20, dx - 20:dx + 20] = dealer
    return img


def reader(tmp_path):
    return SeatReader({"dealer_templates": [str(tmp_path / "d.png")],
                       "cardback_templates": [str(tmp_path / "c.png")], "ref_text_height": 0})


@pytest.mark.parametrize("n_seats,occupied,btn,cards,bets,pos,opp", [
    (6, [0, 1, 2, 3, 4, 5], 0, [1, 2], [], "BTN", 2),
    (6, [0, 1, 2, 3, 4, 5], 3, [1, 2, 4, 5], [4, 5], "UTG", 4),
    (9, [0, 2, 3, 5, 7, 8], 2, [3, 7], [7, 8], "CO", 2),        # 9 places, 6 joueurs, sièges vides
    (9, [0, 4, 6], 6, [4, 6], [], "SB", 2),                    # 9 places, 3 joueurs
    (5, [0, 2, 3], 3, [2], [2], "SB", 1),                      # 5 places, 3 joueurs
])
def test_any_table_size(tmp_path, n_seats, occupied, btn, cards, bets, pos, opp):
    dealer, card = _templates(tmp_path)
    img = make_table(dealer, card, n_seats, occupied, btn, cards, bets)
    assert len(find_players(img)) == len(occupied)
    facts = reader(tmp_path).read(img)
    assert (facts.position, facts.opponents) == (pos, opp)


def test_not_ready_without_templates(tmp_path):
    r = SeatReader({"dealer_templates": [str(tmp_path / "x.png")], "cardback_templates": []})
    assert not r.ready and r.read(np.zeros((10, 10, 3), np.uint8)) is None
