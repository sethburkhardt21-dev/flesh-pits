"""Unit tests for EXP-FP-0126's adaptive terminal detector.

Additive; the canonical ECR module is untouched.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_ecr_detector import AdaptiveTerminalECR
from attention_ecr_repro import ReturnConditionedEpisodicArbitrator

CH = ["a", "b", "c", "d"]


def make(detector="relmax"):
    return AdaptiveTerminalECR(
        CH, detector=detector,
        branch_channels={"a", "b"}, max_steps=10)


def feed(arb, stream):
    """stream: list of (winner, reward). Returns list of terminal flags."""
    flags = []
    for winner, reward in stream:
        arb._pending_winner = winner
        arb.update_gains({winner: reward} if winner else {})
        flags.append(arb.gain_history[-1]["terminal"])
    return flags


def test_d0_hardcoded_misfires_on_098():
    arb = make("hardcoded")
    flags = feed(arb, [("c", -0.02), ("c", 0.98)])
    assert flags == [False, False], flags  # the 0120 failure mode


def test_d0_fires_at_10():
    arb = make("hardcoded")
    assert feed(arb, [("a", 1.0)]) == [True]


def test_d1_floor_fires_on_098():
    arb = make("floor")
    flags = feed(arb, [("c", -0.02), ("c", 0.98)])
    assert flags == [False, True], flags


def test_d1_shaping_stays_shaping():
    arb = make("floor")
    flags = feed(arb, [("c", 0.02), ("c", 0.02)])
    assert flags == [False, False], flags


def test_d3_relmax_fires_on_098():
    arb = make("relmax")
    flags = feed(arb, [("c", -0.02), ("c", 0.98)])
    assert flags == [False, True], flags
    assert arb._r_big == 0.98


def test_d3_delayed_reward_stream():
    # shaping ticks are not terminal; the 1.0 terminal is.
    arb = make("relmax")
    stream = [("c", 0.02)] * 5 + [("c", 1.0)] + [("c", 0.02)] * 3
    flags = feed(arb, stream)
    assert flags == [False] * 5 + [True] + [False] * 3, flags


def test_d3_second_goal_still_terminal():
    arb = make("relmax")
    flags = feed(arb, [("c", 0.98), ("c", -0.02), ("c", 0.98)])
    assert flags == [True, False, True], flags


def test_d2_baseline_k_times_mean():
    arb = make("baseline")
    # 0.02 < 0.1 floor -> shaping; 1.0 > 0.1 -> terminal;
    # after a 1.0, shaping 0.02 < max(2*~0.1, 0.1)=0.2 -> shaping.
    flags = feed(arb, [("c", 0.02), ("c", 1.0), ("c", 0.02)])
    assert flags == [False, True, False], flags


def test_d0_gain_equivalence_with_parent():
    # D0 must be behaviorally identical to the parent class on gains:
    # same stream -> identical gains at every step.
    stream = [("a", 0.0), ("c", 0.02), ("d", 0.02), ("a", 0.0),
              ("c", 1.0), ("c", 0.02), ("d", 0.0)]
    parent = ReturnConditionedEpisodicArbitrator(
        CH, branch_channels={"a", "b"}, max_steps=10)
    child = make("hardcoded")
    for winner, reward in stream:
        parent._pending_winner = winner
        child._pending_winner = winner
        g_p = parent.update_gains({winner: reward} if winner else {})
        g_c = child.update_gains({winner: reward} if winner else {})
        assert g_p == g_c, (g_p, g_c)


def test_frozen_touches_nothing():
    arb = make("relmax")
    arb.frozen = True
    arb._pending_winner = "c"
    arb.update_gains({"c": 0.98})
    assert arb._n_pos == 0 and arb._r_big == 0.0
    assert dict(arb.gains) == {c: 1.0 for c in CH}


def test_invalid_detector_rejected():
    try:
        make("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_detector_stats_post_classification():
    # stats update AFTER classification: a first-seen 0.98 classifies as
    # terminal even though it then becomes r_big=0.98.
    arb = make("relmax")
    arb._pending_winner = "c"
    arb.update_gains({"c": 0.98})
    assert arb.gain_history[-1]["terminal"] is True
    assert arb._r_big == 0.98


def make_guarded():
    return AdaptiveTerminalECR(
        CH, detector="relmax", progress_guard=True,
        branch_channels={"a", "b"}, max_steps=10)


def test_guard_skips_progress_without_shaping():
    arb = make_guarded()
    # terminal tick with zero shaping receipts: branch pinned, no progress pin
    arb._pending_winner = "a"
    arb.update_gains({"a": 0.98})
    h = arb.gain_history[-1]
    assert h["terminal"] is True
    assert h["branch_credit"] == "a"
    assert h["progress_credit"] is None
    assert h["progress_guarded"] is True
    assert arb.gains["a"] == arb.gain_cap
    for c in ("b", "c", "d"):
        assert arb.gains[c] == 1.0, (c, arb.gains[c])


def test_guard_inert_with_shaping():
    arb = make_guarded()
    feed(arb, [("c", 0.02), ("c", 0.02), ("a", 0.98)])
    h = arb.gain_history[-1]
    assert h["terminal"] is True
    assert h.get("progress_guarded", False) is False
    # progress = most shaping receipts = "c" -> capped like the parent
    assert h["progress_credit"] == "c"
    assert arb.gains["c"] == arb.gain_cap


def test_guard_off_matches_parent():
    # progress_guard=False must be gain-identical to the parent class
    # on a stream with a terminal tick and no shaping.
    stream = [("a", 0.0), ("d", 0.0), ("a", 0.98), ("c", -0.02)]
    parent = ReturnConditionedEpisodicArbitrator(
        CH, branch_channels={"a", "b"}, max_steps=10)
    child = AdaptiveTerminalECR(
        CH, detector="hardcoded", progress_guard=False,
        branch_channels={"a", "b"}, max_steps=10)
    for winner, reward in stream:
        parent._pending_winner = winner
        child._pending_winner = winner
        g_p = parent.update_gains({winner: reward} if winner else {})
        g_c = child.update_gains({winner: reward} if winner else {})
        assert g_p == g_c, (g_p, g_c)
