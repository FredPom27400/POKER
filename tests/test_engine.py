import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anthropic
try:
    import httpx2 as httpx  # anthropic >= 1.0
except ImportError:  # pragma: no cover
    import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_assistant.claude_client import Advice, ClaudeAdvisor  # noqa: E402
from poker_assistant.config import DEFAULTS, _merge  # noqa: E402
from poker_assistant.engine import Engine  # noqa: E402
from poker_assistant.journal import DecisionLogger  # noqa: E402
from tests.synthetic import BAR, make_table  # noqa: E402


def make_cfg(**dedupe):
    cfg = _merge(DEFAULTS, {"action_bar": BAR, "dedupe": dict(dedupe, min_seconds_between_calls=0)})
    cfg["detection"].update(template_turn=None, template_wait=None)
    return cfg


def run_frames(engine, frames):
    for f in frames:
        engine.grab = lambda f=f: f
        engine.step()
        time.sleep(0.02)
    engine._pool.shutdown(wait=True)
    engine._drain_results()


def test_engine_calls_once_per_turn(tmp_path):
    calls = []

    def ask(img):
        calls.append(img)
        return Advice(True, "ACTION : Call\nRAISON : test", "fake")

    eng = Engine(make_cfg(), None, ask, DecisionLogger(tmp_path))
    wait1, turn1 = make_table(False, seed=1, board=0), make_table(True, seed=1, board=0, timer=15)
    turn1b = make_table(True, seed=1, board=0, timer=9)
    wait2, turn2 = make_table(False, seed=1, board=3), make_table(True, seed=1, board=3)
    run_frames(eng, [wait1, wait1, turn1, turn1, turn1b, turn1b, turn1b, wait2, wait2, turn2, turn2, turn2])
    assert len(calls) == 2
    day = next(tmp_path.iterdir())
    assert len(list(day.glob("*.png"))) == 2
    assert (day / "decisions.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_engine_preaction_bar_blinks_one_frame(tmp_path):
    """Les adversaires jouent en pré-action : la barre ne disparaît qu'une frame."""
    calls = []
    eng = Engine(make_cfg(), None, lambda img: calls.append(1) or Advice(True, "ok"), DecisionLogger(tmp_path))
    preflop, flop = make_table(True, seed=1, board=0), make_table(True, seed=1, board=3)
    run_frames(eng, [preflop] * 3 + [make_table(False, seed=1, board=0)] + [flop] * 4)
    assert len(calls) == 2


def test_engine_preaction_bar_never_disappears(tmp_path):
    calls = []
    eng = Engine(make_cfg(), None, lambda img: calls.append(1) or Advice(True, "ok"), DecisionLogger(tmp_path))
    preflop, flop = make_table(True, seed=1, board=0), make_table(True, seed=1, board=3)
    run_frames(eng, [preflop] * 4 + [flop] * 5)
    assert len(calls) == 2


def test_engine_survives_exceptions(tmp_path):
    n = []

    def ask(img):
        n.append(1)
        raise RuntimeError("boom")

    eng = Engine(make_cfg(retry_delay=0), None, ask, DecisionLogger(tmp_path))
    run_frames(eng, [make_table(True)] * 2)
    assert n == [1]
    assert eng.tracker.retry_at is not None


class FakeStream:
    def __init__(self, resp):
        self.resp = resp
        self.text_stream = [b.text for b in resp.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.resp


class FakeMessages:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    @property
    def kwargs(self):
        return self.calls[-1]

    def stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.behaviour(kw))


def advisor_with(behaviour, **cfg):
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")
    adv = ClaudeAdvisor(dict(DEFAULTS["claude"], **cfg))
    msgs = FakeMessages(behaviour)
    adv.client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return adv, msgs


def ok_resp(text="ACTION : Fold\nRAISON : x", stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, model="claude-opus-5-5",
                           usage=SimpleNamespace(output_tokens=42, speed="fast"),
                           content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)])


REQ = httpx.Request("POST", "https://api.anthropic.com")


def test_advisor_success_request_shape():
    partial = []
    adv, msgs = advisor_with(lambda kw: ok_resp("**ACTION :** Fold\nRAISON : x"))
    a = adv.ask(make_table(True, w=2400, h=1350), on_text=partial.append)
    assert a.ok and a.text.startswith("ACTION : Fold") and partial
    img = msgs.kwargs["messages"][0]["content"][0]
    assert img["type"] == "image" and img["source"]["media_type"] == "image/png"
    assert msgs.kwargs["fallbacks"] == "default"
    assert msgs.kwargs["speed"] == "fast" and "fast-mode-2026-02-01" in msgs.kwargs["betas"]
    assert msgs.kwargs["output_config"] == {"effort": "low"}


def test_thinking_off_per_model():
    adv, msgs = advisor_with(lambda kw: ok_resp(), model="claude-sonnet-5-5", thinking="off")
    a = adv.ask(make_table(True))
    assert a.ok and a.out_tokens == 42
    kw = msgs.kwargs
    assert kw["thinking"] == {"type": "between_tools"}
    assert "speed" not in kw and "fallbacks" not in kw     # pas de mode rapide ni de repli sur Sonnet sans réflexion

    adv, msgs = advisor_with(lambda kw: ok_resp(), model="claude-haiku-5-5", thinking="off")
    adv.ask(make_table(True))
    assert msgs.kwargs["thinking"] == {"type": "disabled"} and "fallbacks" not in msgs.kwargs

    adv, msgs = advisor_with(lambda kw: ok_resp(), thinking="off")   # Opus : impossible, ignoré
    adv.ask(make_table(True))
    assert "thinking" not in msgs.kwargs and msgs.kwargs["speed"] == "fast"


def test_timing_text():
    from poker_assistant.claude_client import Advice as A
    t = A(True, "x", seconds=9.8, first_text=8.9, out_tokens=640, fast=True).timing()
    assert t == "9.8 s · dont 8.9 s avant le 1er mot · 640 tokens générés · mode rapide"


def test_advisor_normal_speed():
    adv, msgs = advisor_with(lambda kw: ok_resp(), speed="normal")
    assert adv.ask(make_table(True)).ok
    assert "speed" not in msgs.kwargs and "fast-mode-2026-02-01" not in msgs.kwargs["betas"]


def test_fast_mode_refused_falls_back_to_normal():
    def behaviour(kw):
        if kw.get("speed") == "fast":
            raise anthropic.BadRequestError("speed not supported", response=httpx.Response(400, request=REQ), body=None)
        return ok_resp()

    adv, msgs = advisor_with(behaviour)
    assert adv.ask(make_table(True)).ok
    assert adv.fast is False                     # coupé pour la suite de la session
    assert adv.ask(make_table(True)).ok
    assert len(msgs.calls) == 3


def test_fast_mode_rate_limited_retries_normal_once():
    def behaviour(kw):
        if kw.get("speed") == "fast":
            raise anthropic.RateLimitError("rl", response=httpx.Response(429, request=REQ), body=None)
        return ok_resp()

    adv, _ = advisor_with(behaviour)
    assert adv.ask(make_table(True)).ok
    assert adv.fast is True                      # simple saturation : on garde le mode rapide


def test_advisor_timeout_and_refusal():
    def timeout(kw):
        raise anthropic.APITimeoutError(request=REQ)

    adv, _ = advisor_with(timeout)
    a = adv.ask(make_table(True))
    assert not a.ok and "timeout" in a.error.lower()

    def rate_limit(kw):
        raise anthropic.RateLimitError("rl", response=httpx.Response(429, request=REQ), body=None)

    adv, _ = advisor_with(rate_limit, speed="normal")
    assert "429" in adv.ask(make_table(True)).error

    adv, _ = advisor_with(lambda kw: ok_resp("", stop="refusal"))
    assert not adv.ask(make_table(True)).ok


def test_engine_streams_partial_text(tmp_path):
    import queue as q
    events = q.Queue()

    def ask(img, on_text=None):
        on_text("ACTION : Call")
        return Advice(True, "ACTION : Call\nRAISON : x", seconds=1.2)

    eng = Engine(make_cfg(), None, ask, DecisionLogger(tmp_path), events)
    run_frames(eng, [make_table(True)] * 2)
    got = []
    while not events.empty():
        got.append(events.get())
    assert ("partial", "ACTION : Call") in got


def test_history_sent_on_next_call(tmp_path):
    seen = []

    def ask(img, on_text=None, history=None):
        seen.append(list(history or []))
        return Advice(True, "LECTURE : 6♠6♦ | 8♦3♣K♠ | bouton | 1 | 3 BB\nACTION : Call 3 BB\nRAISON : x")

    eng = Engine(make_cfg(), None, ask, DecisionLogger(tmp_path))
    flop, turn = make_table(True, seed=1, board=3), make_table(True, seed=1, board=4)
    run_frames(eng, [flop] * 2 + [make_table(False, seed=1, board=3)] * 3 + [turn] * 2)
    assert len(seen) == 2 and seen[0] == []
    assert "6♠6♦" in seen[1][0] and "Call 3 BB" in seen[1][0]


def test_history_expires(tmp_path):
    eng = Engine(make_cfg(), None, lambda img, history=None: Advice(True, "x"), DecisionLogger(tmp_path))
    eng._remember("LECTURE : A♠K♠ | - | UTG | 4 | 1 BB\nACTION : Raise to 2.5 BB")
    assert len(eng._recent_history()) == 1
    eng._history = [(t - 1000, h) for t, h in eng._history]
    assert eng._recent_history() == []


def test_prompt_and_parsing():
    from poker_assistant import claude_client, pretty
    assert "Sens du jeu" in claude_client.SYSTEM_PROMPT and "SIEGES" in claude_client.SYSTEM_PROMPT
    assert claude_client.user_text(None) == claude_client.USER_TEXT
    assert "même main" in claude_client.user_text(["il y a 10 s : X → Call"])
    f = pretty.parse_fields("LECTURE : Q♣2♥ | - | SB | 1 | 0,5 BB\nACTION : Call\nRAISON : r\nNOTE : n")
    assert (f["lecture"], f["action"], f["reason"], f["notes"]) == ("Q♣2♥ | - | SB | 1 | 0,5 BB", "Call", "r", ["n"])


def test_hidden_table_skips_call(tmp_path):
    import cv2
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), make_table(True, seed=1))
    cfg = make_cfg()
    cfg["detection"].update(table_ref=str(ref), min_table_similarity=0.6)
    calls = []
    eng = Engine(cfg, None, lambda img: calls.append(1) or Advice(True, "ok"), DecisionLogger(tmp_path))
    hidden = make_table(True, seed=1)
    hidden[: int(hidden.shape[0] * 0.85)] = (235, 235, 235)   # fenêtre claire devant la table
    visible = make_table(True, seed=1)
    calls_img = []
    eng.ask = lambda img: calls_img.append(img) or Advice(True, "ok")
    run_frames(eng, [hidden] * 4 + [visible] * 3)              # puis table de nouveau visible
    assert len(calls_img) == 1 and (calls_img[0] == visible).all()


def test_eval_parsing_and_cases():
    import yaml
    from eval_lecture import parse_lecture
    assert parse_lecture("7♥ J♦ | - | BTN | 2 | 1 BB") == {"position": "BTN", "adversaires": 2, "a_payer": 1}
    assert parse_lecture("x | - | BB (bouton chez X) | 1 adversaire | 0 (CHECK)")["position"] == "BB"
    assert parse_lecture("x | - | Cutoff (le bouton est à gauche) | 3 | 1 BB")["position"] == "CO"
    assert parse_lecture("x | - | petite blinde | 3 | 10,5 BB")["a_payer"] == 10.5
    spec = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "eval" / "cases.yaml", encoding="utf-8"))
    assert all(isinstance(c["fichier"], str) and "_" in c["fichier"] for c in spec["cas"])


def test_lecture_summary():
    from poker_assistant.pretty import lecture_summary
    assert lecture_summary("K♦ 9♠ | préflop, pas de board | BTN | 1 adversaire (X) | 7,5 BB") == \
        "K♦ 9♠  ·  préflop  ·  BTN  ·  1 adversaire (X)  ·  à payer 7,5 BB"
    assert lecture_summary("A♣ 8♣ | J♦ Q♦ 4♦ | BB | 1 | 0") == "A♣ 8♣  ·  J♦ Q♦ 4♦  ·  BB  ·  1 adv.  ·  rien à payer"


def test_style_in_system_prompt():
    adv, msgs = advisor_with(lambda kw: ok_resp(), style="tag")
    adv.ask(make_table(True))
    assert "serré-agressif" in msgs.kwargs["system"]
    adv, msgs = advisor_with(lambda kw: ok_resp(), style="equilibre")
    adv.ask(make_table(True))
    assert "Style de jeu" not in msgs.kwargs["system"]


def test_shortcuts_in_prompt_and_keys_parsed():
    from poker_assistant.claude_client import system_prompt
    from poker_assistant.config import DEFAULTS
    from poker_assistant.pretty import parse_fields
    p = system_prompt("tag", DEFAULTS["claude"]["shortcuts"])
    assert "x2.5 (F4)" in p and "65% (F5)" in p and "TOUCHES" in p
    assert "TOUCHES" not in system_prompt("tag", None)
    f = parse_fields("ACTION : Raise to 2,5 BB\nTOUCHES : F4 puis F3\nRAISON : r")
    assert f["keys"] == "F4 puis F3" and f["notes"] == []


def test_check_never_on_fold_key():
    from poker_assistant.claude_client import fix_keys
    from poker_assistant.config import DEFAULTS
    sc = DEFAULTS["claude"]["shortcuts"]
    assert "TOUCHES : F2" in fix_keys("ACTION : Check\nTOUCHES : F1\nRAISON : x", sc)
    assert "TOUCHES : F1" in fix_keys("ACTION : Fold\nTOUCHES : F1", sc)
    assert "TOUCHES : F4 puis F3" in fix_keys("ACTION : Bet 3 BB\nTOUCHES : F4 puis F3", sc)


def test_self_correction_last_action_wins():
    from poker_assistant.pretty import lecture_summary, parse_fields
    f = parse_fields("ACTION : Fold\nTOUCHES : F1\nRAISON : r\nCorrection : en fait si.\n"
                     "ACTION finale : Raise to 2,5 BB\nTOUCHES : F4 puis F3\nNOTE : aucune.")
    assert f["action"] == "Raise to 2,5 BB" and f["keys"] == "F4 puis F3"
    assert f["notes"] == ["Correction : en fait si."]
    assert lecture_summary("A♠ 9♦ | préflop | UTG | 4 | 1 BB à payer").endswith("à payer 1 BB")
