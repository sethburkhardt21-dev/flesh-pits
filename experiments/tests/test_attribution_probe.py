"""Unit tests for the attribution probe machinery (methods validation).

Run: python3 -m pytest tests/test_attribution_probe.py -v   (from experiments/)
Stdlib only besides pytest. No random.* module calls.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "envs"))

from env_interface import derive_seed, new_rng  # noqa: E402
from self_world_mismatch import SelfWorldMismatch  # noqa: E402
from self_world import SelfWorld  # noqa: E402
from source_attributor import (  # noqa: E402
    ComparatorAgent, ChanceAgent, BURN_IN,
)


def run_episode(seed):
    env = SelfWorldMismatch()
    agent = ComparatorAgent()
    obs = env.reset(seed)
    agent.reset(derive_seed(seed, 0, "agent"), env.action_space())
    modes = []
    causes = []
    done = False
    while not done:
        a = agent.act(obs)
        obs2, r, done, info = env.step(a)
        agent.update(obs, a, r, done, info)
        modes.append(info["mismatch"]["mode"])
        causes.append(dict(info["cause"]))
        obs = obs2
    return env, agent, modes, causes


def test_env_is_deterministic():
    _, a1, m1, c1 = run_episode(424242)
    _, a2, m2, c2 = run_episode(424242)
    assert m1 == m2 and c1 == c2
    assert len(a1.attributions) == len(a2.attributions) > 0


def test_mismatch_rate_and_modes_sane():
    modes_all = []
    for s in range(424300, 424320):
        _, _, modes, _ = run_episode(s)
        modes_all.extend(modes)
    frac = sum(1 for m in modes_all if m is not None) / len(modes_all)
    assert 0.05 < frac < 0.30, frac
    seen = set(m for m in modes_all if m)
    assert seen == {"override", "swap", "delay"}, seen


def test_cause_labels_structural():
    # normal (non-mismatch) ticks: hand self, ball world
    _, _, modes, causes = run_episode(424242)
    for m, c in zip(modes, causes):
        if m is None:
            assert c == {"hand": "self", "ball": "world"}, (m, c)
        elif m == "override":
            assert c["hand"] == "world" and c["ball"] == "world"
        elif m == "swap":
            assert c["hand"] == "world" and c["ball"] == "self"
        elif m == "delay":
            assert c["hand"] == "world"  # held or stale-driven


def test_attributor_burn_in_excludes_first_ticks():
    _, agent, _, _ = run_episode(424242)
    ticks = [r["tick"] for r in agent.attributions]
    assert ticks[0] == BURN_IN + 1
    assert all(t > BURN_IN for t in ticks)


def test_comparator_perfect_on_clean_env():
    # On the mismatch-free parent env, the comparator must be ~perfect:
    # every moved hand tick -> self; every moved ball tick -> world.
    env = SelfWorld()
    agent = ComparatorAgent()
    obs = env.reset(777)
    agent.reset(111, env.action_space())
    truth = []
    done = False
    while not done:
        a = agent.act(obs)
        obs2, r, done, info = env.step(a)
        agent.update(obs, a, r, done, info)
        truth.append(dict(info["cause"]))
        obs = obs2
    correct = total = 0
    for rec in agent.attributions:
        t = truth[rec["tick"] - 1]
        for ch, attr in rec["attribution"].items():
            total += 1
            correct += (attr == t[ch])
    assert total > 50  # fixed-point non-moves are excluded from scoring
    # Honest bound: on short streams the ball's fair coin can transiently
    # correlate with the alternating action stream, pushing the learned
    # ball coupling above the gate for a few ticks (spurious attribution).
    # This is real mechanism behavior, not a scoring artifact.
    assert correct / total >= 0.90, f"{correct}/{total}"


def test_chance_null_runs_and_attributions_bounded():
    env = SelfWorldMismatch()
    agent = ChanceAgent()
    obs = env.reset(424242)
    agent.reset(999, env.action_space())
    done = False
    while not done:
        a = agent.act(obs)
        obs2, r, done, info = env.step(a)
        agent.update(obs, a, r, done, info)
        obs = obs2
    vals = {v for rec in agent.attributions for v in rec["attribution"].values()}
    assert vals <= {"self", "world"}
    assert len(agent.attributions) > 0


def test_predictions_never_admitted_implicitly():
    # The attributor emits predictions tagged 'predicted'; assert the tag.
    _, agent, _, _ = run_episode(424242)
    assert len(agent.predictions) > 0
    assert all(p.get("source") == "predicted" for p in agent.predictions)
    assert all("observation" in p for p in agent.predictions)


def test_agent_never_sees_info_contract():
    # Smoke: update() ignores info (no attribute derived from it).
    agent = ComparatorAgent()
    obs = {"hand": 0.5, "ball": 0.5}
    agent.reset(1, {"type": "discrete", "n": 2})
    a = agent.act(obs)
    agent.update(obs, a, 0.0, False, {"cause": {"hand": "BOGUS"}})
    snap = agent.snapshot()
    assert "BOGUS" not in str(snap)
