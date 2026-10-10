import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_assistant.config import DEFAULTS, load_config  # noqa: E402
from poker_assistant.detection import Debouncer, TurnDetector, crop_bar  # noqa: E402
from poker_assistant.state import TurnTracker, change_ratio, signature  # noqa: E402
from tests.synthetic import BAR, make_screen, make_table  # noqa: E402
import test_detection  # noqa: E402

DET = dict(DEFAULTS["detection"], template_turn=None, template_wait=None)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("turn", [True, False])
def test_color_detection(turn, seed):
    d = TurnDetector(DET, BAR).detect(make_table(turn, seed=seed))
    assert d.is_turn is turn
    if turn:
        assert d.buttons == 3


@pytest.mark.parametrize("gap", [2, 4, 8])
def test_tightly_packed_buttons(gap):
    """Bug réel (table Expresso) : boutons séparés de quelques pixels -> fusionnés en 1."""
    d = TurnDetector(DET, BAR).detect(make_table(True, gap=gap))
    assert d.is_turn and d.buttons == 3


@pytest.mark.parametrize("size", [(960, 540), (1920, 1080)])
def test_resolution_independent(size):
    det = TurnDetector(DET, BAR)
    assert det.detect(make_table(True, w=size[0], h=size[1])).is_turn
    assert not det.detect(make_table(False, w=size[0], h=size[1])).is_turn


def test_template_hybrid(tmp_path):
    turn_tpl = crop_bar(make_table(True, seed=1, timer=12), BAR)
    wait_tpl = crop_bar(make_table(False, seed=1), BAR)
    cv2.imwrite(str(tmp_path / "t.png"), turn_tpl)
    cv2.imwrite(str(tmp_path / "w.png"), wait_tpl)
    cfg = dict(DET, template_turn=str(tmp_path / "t.png"), template_wait=str(tmp_path / "w.png"))
    det = TurnDetector(cfg, BAR)
    on = det.detect(make_table(True, seed=7, timer=3))
    off = det.detect(make_table(False, seed=7))
    assert on.is_turn and on.turn_score > on.wait_score
    assert not off.is_turn and off.wait_score > off.turn_score


def test_template_vetoes_red_bar_not_matching(tmp_path):
    # Barre entièrement rouge (ex. bannière) : la couleur dit oui, le template dit non.
    table = make_table(False)
    x0, y0 = int(BAR["x"] * table.shape[1]), int(BAR["y"] * table.shape[0])
    table[y0 + 5:y0 + 40, x0 + 5:x0 + 200] = (35, 35, 200)
    table[y0 + 45:y0 + 80, x0 + 300:x0 + 600] = (35, 35, 200)
    cv2.imwrite(str(tmp_path / "w.png"), crop_bar(make_table(False), BAR))
    cv2.imwrite(str(tmp_path / "t.png"), crop_bar(make_table(True), BAR))
    det = TurnDetector(dict(DET, template_turn=str(tmp_path / "t.png"),
                            template_wait=str(tmp_path / "w.png")), BAR)
    assert not det.detect(table).is_turn


def test_debouncer():
    db = Debouncer(2)
    assert [db.update(v) for v in [True, False, True, True, False, True, False, False]] == \
           [False, False, False, True, True, True, True, False]


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_tracker_one_call_per_turn():
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    a = signature(make_table(True, seed=1, timer=15), BAR)
    a2 = signature(make_table(True, seed=1, timer=4), BAR)   # seul le timer change
    b = signature(make_table(True, seed=2, board=0), BAR)   # nouvelle main (préflop)

    assert tr.should_call(True, a)
    tr.on_result(True)
    for i in range(10):
        clk.t += 0.5
        assert not tr.should_call(True, a2 if i % 2 else a)
    for _ in range(3):                  # les adversaires jouent
        clk.t += 0.5
        assert not tr.should_call(False, a)
    clk.t += 0.5
    assert tr.should_call(True, b)     # nouveau tour, nouvelle table
    tr.on_result(True)
    clk.t += 0.5
    assert not tr.should_call(False, b)
    for _ in range(6):                  # barre qui clignote, table identique
        clk.t += 0.5
        assert not tr.should_call(True, b)


def _run_turn(tr, clk, tables):
    calls = []
    for i, t in enumerate(tables):
        clk.t += 0.5
        if tr.should_call(True, signature(t, BAR)):
            calls.append((i, tr.reason))
            tr.on_result(True)
    return calls


def test_seat_timer_ring_does_not_trigger():
    """Bug réel : le chrono autour de l'avatar se vide -> « table modifiée »."""
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    frames = [make_table(True, seed=1, ring=1 - i / 30) for i in range(30)]   # 15 s
    assert _run_turn(tr, clk, frames) == [(0, "nouveau tour")]


def test_seat_timer_digits_do_not_trigger():
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    frames = [make_table(True, seed=1, seat_clock=20 - i // 2, ring=1 - i / 40) for i in range(40)]
    assert _run_turn(tr, clk, frames) == [(0, "nouveau tour")]


def test_preaction_jump_detected_despite_timer():
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    frames = [make_table(True, seed=1, board=0, seat_clock=20 - i // 2, ring=1 - i / 20) for i in range(12)]
    frames += [make_table(True, seed=1, board=3, seat_clock=20 - i // 2, ring=1 - i / 20) for i in range(6)]
    calls = _run_turn(tr, clk, frames)
    assert [c[0] for c in calls] == [0, 14]


def test_signature_sensitivity():
    base = signature(make_table(True, seed=1, board=3, timer=15), BAR)
    assert change_ratio(base, signature(make_table(True, seed=1, board=3, timer=2), BAR)) == 0
    thr = DEFAULTS["dedupe"]["change_ratio"]
    assert change_ratio(base, signature(make_table(True, seed=1, board=4), BAR)) > thr  # turn
    assert change_ratio(base, signature(make_table(False, seed=1, board=3), BAR)) < thr


def test_tracker_recall_on_table_change():
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    a = signature(make_table(True, seed=1, board=3), BAR)
    b = signature(make_table(True, seed=1, board=4), BAR)
    assert tr.should_call(True, a)
    tr.on_result(True)
    clk.t += 3
    assert not tr.should_call(True, b)  # changement pas encore stable
    clk.t += 0.5
    assert not tr.should_call(True, b)
    clk.t += 0.5
    assert tr.should_call(True, b)
    assert "table" in tr.reason
    tr.on_result(True)
    for _ in range(6):
        clk.t += 0.5
        assert not tr.should_call(True, b)


def test_tracker_recall_on_layout_but_not_hover():
    clk = Clock()
    cfg = dict(DEFAULTS["dedupe"], recall_on_table_change=False)
    tr = TurnTracker(cfg, clock=clk)
    a = signature(make_table(True, seed=1), BAR)
    assert tr.should_call(True, a, (4, 10, 16))       # FOLD / CALL / RAISE
    tr.on_result(True)
    for _ in range(5):                                # souris sur un bouton : il perd son rouge
        clk.t += 0.5
        assert not tr.should_call(True, a, (4, 16))
    for i in range(3):                                # CHECK / BET : nouvelle décision
        clk.t += 0.5
        assert tr.should_call(True, a, (6, 14)) is (i == 2)
    assert "boutons" in tr.reason


def test_tracker_max_calls_per_turn():
    clk = Clock()
    tr = TurnTracker(dict(DEFAULTS["dedupe"], max_calls_per_turn=2), clock=clk)
    calls = 0
    for board in (0, 3, 4, 5):
        sig = signature(make_table(True, seed=1, board=board), BAR)
        for _ in range(4):
            clk.t += 1
            if tr.should_call(True, sig):
                calls += 1
                tr.on_result(True)
    assert calls == 2


def test_tracker_retry_after_error():
    clk = Clock()
    cfg = DEFAULTS["dedupe"]
    tr = TurnTracker(cfg, clock=clk)
    a = signature(make_table(True, seed=1), BAR)
    assert tr.should_call(True, a)
    assert not tr.should_call(True, a)   # appel en cours
    tr.on_result(False)
    clk.t += cfg["retry_delay"] - 0.1
    assert not tr.should_call(True, a)
    clk.t += 0.2
    assert tr.should_call(True, a)
    tr.on_result(False)
    clk.t += cfg["retry_delay"] + 0.1
    assert tr.should_call(True, a)
    tr.on_result(False)                 # retries épuisés
    clk.t += 10
    assert not tr.should_call(True, a)


def test_full_screen_sample_cropping():
    cfg = load_config("/nonexistent.yaml")
    cfg["capture"]["region"] = {"x": 200, "y": 150, "width": 1280, "height": 720}
    screen = make_screen(make_table(True))
    table = test_detection.to_table(screen, cfg["capture"]["region"])
    assert table.shape[:2] == (720, 1280)
    assert TurnDetector(DET, BAR).detect(table).is_turn
    assert test_detection.expected_from_name("montour_01.png") is True
    assert test_detection.expected_from_name("attente_presel.png") is False
    assert np.array_equal(test_detection.to_table(table, cfg["capture"]["region"]), table)


def test_change_during_api_call_is_absorbed():
    """Bug réel : la table bouge pendant que Claude répond (animation, étiquette
    d'action qui s'efface) -> 2e appel « table modifiée » juste après la réponse."""
    clk = Clock()
    tr = TurnTracker(DEFAULTS["dedupe"], clock=clk)
    def table(board, label):
        img = make_table(True, seed=1, board=board)
        if label:  # étiquette « RAISE » d'un adversaire, qui s'efface après quelques secondes
            cv2.rectangle(img, (950, 120), (1110, 170), (0, 140, 255), -1)
        return signature(img, BAR)

    start, settled = table(3, True), table(3, False)
    assert change_ratio(start, settled) > 2 * DEFAULTS["dedupe"]["change_ratio"]
    assert tr.should_call(True, start)
    for sig in [start, settled, settled, settled, settled]:     # pendant l'appel API
        clk.t += 0.5
        assert not tr.should_call(True, sig)
    tr.on_result(True)
    for _ in range(8):
        clk.t += 0.5
        assert not tr.should_call(True, settled)
    # Une vraie nouvelle situation ensuite est toujours détectée.
    turn_card = table(4, False)
    calls = 0
    for _ in range(4):
        clk.t += 0.5
        if tr.should_call(True, turn_card):
            calls += 1
            tr.on_result(True)
    assert calls == 1
