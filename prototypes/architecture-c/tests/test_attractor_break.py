"""EXP-FP-0011 — unit tests for the attractor-break intervention variants.

Behavioral, not self-confirming: each test asserts a difference the
intervention must produce vs the canonical path it subclasses.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_C = os.path.dirname(_HERE)
for _p in (_HERE, _ARCH_C):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_attractor_break import (
    RepeatPenaltyArbitrator, RepeatPenaltyCueArbitrator,
    LiftedFloorArbitrator, LiftedFloorCueArbitrator,
    ArchCRepeatPenalty, ArchCGainLift, ArchCAttractorBreak,
)
from attention import AttentionArbitrator
from attention_cue import CueIndexedArbitrator

CHANNELS = ["north", "south", "east", "west"]
CONST_STIM = {"north": 1.0, "south": 0.5, "east": 0.5, "west": 0.5}


def _run_ticks(arb, stimuli, n):
    return [arb.arbitrate(dict(stimuli))["winner"] for _ in range(n)]


def test_repeat_penalty_breaks_frozen_winner():
    """Canonical base freezes on a constant-stimulus winner; the penalty
    variant must rotate winners (no channel wins more than WINDOW
    consecutive ticks)."""
    base = AttentionArbitrator(CHANNELS)
    pen = RepeatPenaltyArbitrator(CHANNELS)
    wb = _run_ticks(base, CONST_STIM, 12)
    wp = _run_ticks(pen, CONST_STIM, 12)
    assert len(set(wb)) == 1, f"expected frozen base winner, got {wb}"
    assert len(set(wp)) > 1, f"penalty did not rotate: {wp}"
    streak = 1
    for a, b in zip(wp, wp[1:]):
        streak = streak + 1 if a == b else 1
        assert streak <= RepeatPenaltyArbitrator.WINDOW, \
            f"streak {streak} exceeded WINDOW: {wp}"


def test_repeat_penalty_formula_matches_history():
    """The penalty is a symmetric function of win recency only: for every
    channel, penalty == PENALTY * (wins in last WINDOW). (Channel-order
    tie-break asymmetry lives in the canonical base class, not here.)"""
    p = RepeatPenaltyArbitrator(CHANNELS)
    for _ in range(6):
        d = p.arbitrate(dict(CONST_STIM))
    hist = p._win_history[-p.WINDOW - 1:-1]  # window as seen by that tick
    for c in CHANNELS:
        assert d["repeat_penalty"][c] == p.PENALTY * hist.count(c), \
            (c, d["repeat_penalty"], hist)


def test_repeat_penalty_recorded_and_deterministic():
    p = RepeatPenaltyArbitrator(CHANNELS)
    d1 = p.arbitrate(dict(CONST_STIM))
    assert "repeat_penalty" in d1
    assert d1["repeat_penalty"] == {c: 0.0 for c in CHANNELS}
    p2 = RepeatPenaltyArbitrator(CHANNELS)
    d2 = p2.arbitrate(dict(CONST_STIM))
    assert d1["winner"] == d2["winner"]  # deterministic from same start


def test_gain_floor_lift_holds_under_punishment():
    """Sustained negative utility: canonical floors at 0.01, lift holds
    at 0.5."""
    base = AttentionArbitrator(CHANNELS)
    lift = LiftedFloorArbitrator(CHANNELS)
    for _ in range(50):
        base.update_gains({c: -1.0 for c in CHANNELS}, reward_baseline=0.0)
        lift.update_gains({c: -1.0 for c in CHANNELS}, reward_baseline=0.0)
    assert min(base.gains.values()) == 0.01, base.gains
    assert min(lift.gains.values()) == 0.5, lift.gains
    # identical when utility is positive (only the floor changed)
    base2 = AttentionArbitrator(CHANNELS)
    lift2 = LiftedFloorArbitrator(CHANNELS)
    for _ in range(10):
        base2.update_gains({c: 0.3 for c in CHANNELS}, reward_baseline=0.0)
        lift2.update_gains({c: 0.3 for c in CHANNELS}, reward_baseline=0.0)
    assert base2.gains == lift2.gains


def test_gain_floor_rejects_invalid():
    try:
        LiftedFloorArbitrator(CHANNELS, gain_floor=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("gain_floor=0.0 must be rejected")


def test_cue_variants_cooperate():
    """MRO sanity: penalty + cue-indexing compose; per-context gains;
    penalty still rotates; floor-lift still floors per context."""
    pen = RepeatPenaltyCueArbitrator(CHANNELS)
    pen.set_context(0)
    winners = _run_ticks(pen, CONST_STIM, 12)
    assert len(set(winners)) > 1
    pen.set_context(1)
    assert pen.gains == {c: 1.0 for c in CHANNELS}  # fresh context vector

    lift = LiftedFloorCueArbitrator(CHANNELS)
    lift.set_context(7)
    for _ in range(50):
        lift.update_gains({c: -1.0 for c in CHANNELS}, reward_baseline=0.0)
    assert min(lift.gains.values()) == 0.5
    assert lift.current_context == 7


def _mini_config(context_fn):
    space = {"cue": {"type": "categorical", "values": ["a", "b"]},
             "x": {"type": "scalar"}}
    cfg = {"channels": ["a0", "a1"],
           "channel_to_action": {"a0": 0, "a1": 1},
           "observation_space": space, "env_name": "test_env",
           "context_fn": context_fn, "gain_lr": 0.15, "theta": 0.45,
           "kappa": 0.1, "agent_seed": 42,
           "predictor_kwargs": {"eta0": 0.005, "etaD": 0.002,
                                "eta_r": 0.05, "etaR": 0.10,
                                "precision_kind": "estimated",
                                "max_contexts": 32, "precision_window": 50},
           "memory_enabled": True, "frozen_predictor": False,
           "frozen_gains": False}
    return cfg


def test_variant_agents_wire_correct_arbitrator():
    """Subclass agents wire the variant arbitrators with and without a
    context_fn (control-env path)."""
    a = ArchCRepeatPenalty(_mini_config(None))
    assert isinstance(a.tick.arbitrator, RepeatPenaltyArbitrator)
    b = ArchCGainLift(_mini_config(None))
    assert isinstance(b.tick.arbitrator, LiftedFloorArbitrator)
    assert b.tick.arbitrator.gain_floor == 0.5
    c = ArchCRepeatPenalty(_mini_config(lambda obs: 0))
    assert isinstance(c.tick.arbitrator, RepeatPenaltyCueArbitrator)
    d = ArchCGainLift(_mini_config(lambda obs: 0))
    assert isinstance(d.tick.arbitrator, LiftedFloorCueArbitrator)
    # base class unaffected: plain ArchCAttractorBreak wires canonicals
    e = ArchCAttractorBreak(_mini_config(None))
    assert type(e.tick.arbitrator) is AttentionArbitrator
    f = ArchCAttractorBreak(_mini_config(lambda obs: 0))
    assert type(f.tick.arbitrator) is CueIndexedArbitrator
