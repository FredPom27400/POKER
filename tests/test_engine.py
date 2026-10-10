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


class FakeMessages:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.kwargs = None

    def create(self, **kw):
        self.kwargs = kw
        return self.behaviour()


def advisor_with(behaviour):
    adv = ClaudeAdvisor.__new__(ClaudeAdvisor)
    adv.cfg = DEFAULTS["claude"]
    msgs = FakeMessages(behaviour)
    adv.client = SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs))
    return adv, msgs


def test_advisor_success_request_shape():
    resp = SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                           content=[SimpleNamespace(type="thinking"),
                                    SimpleNamespace(type="text", text="ACTION : Fold\nRAISON : x")])
    adv, msgs = advisor_with(lambda: resp)
    a = adv.ask(make_table(True, w=2400, h=1350))
    assert a.ok and a.text.startswith("ACTION : Fold")
    img = msgs.kwargs["messages"][0]["content"][0]
    assert img["type"] == "image" and img["source"]["media_type"] == "image/png"
    assert msgs.kwargs["fallbacks"] == "default"
    assert msgs.kwargs["output_config"] == {"effort": "low"}


def test_advisor_timeout_and_refusal():
    def timeout():
        raise anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))

    adv, _ = advisor_with(timeout)
    a = adv.ask(make_table(True))
    assert not a.ok and "timeout" in a.error.lower()

    def rate_limit():
        req = httpx.Request("POST", "https://api.anthropic.com")
        raise anthropic.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)

    adv, _ = advisor_with(rate_limit)
    assert "429" in adv.ask(make_table(True)).error

    adv, _ = advisor_with(lambda: SimpleNamespace(stop_reason="refusal", model="m", content=[]))
    assert not adv.ask(make_table(True)).ok
