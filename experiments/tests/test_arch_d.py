"""Tests for Architecture D. Stdlib unittest."""

import json
import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "envs"))
sys.path.insert(0, os.path.join(BASE, "baselines"))
sys.path.insert(0, os.path.join(BASE, "..", "prototypes"))

from envs import ALL_ENVS, ChangingRule  # noqa: E402
from agent_base import flatten_obs, new_rng  # noqa: E402
from predictor import LinearPredictor  # noqa: E402
from arch_d import ArchD  # noqa: E402


def run_steps(env, agent, seed, n=60):
    obs = env.reset(seed)
    agent.reset(seed ^ 0x9E3779B9, env.action_space())
    total, done, i = 0.0, False, 0
    while not done and i < n:
        a = agent.act(obs)
        obs2, r, done, info = env.step(a)
        agent.update(obs, a, r, done, info)
        total += r
        obs = obs2
        i += 1
    return total


class TestArchD(unittest.TestCase):
    def test_runs_all_envs_workspace_bounded(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env, ag = cls(), ArchD()
                run_steps(env, ag, 21, n=40)
                self.assertLessEqual(len(ag.workspace_contents()), 4)
                for kind, bid, _ in ag.workspace_contents():
                    self.assertIn(kind, ("percept", "episode"))
                    self.assertTrue(bid >= 0)

    def test_episodic_store_admits(self):
        env, ag = ChangingRule(), ArchD()
        run_steps(env, ag, 21, n=40)
        self.assertGreater(ag.memory_stats()["store_size"], 0,
                           "store admitted nothing on a changing task")

    def test_determinism(self):
        for cls in ALL_ENVS[:2]:
            with self.subTest(env=cls.NAME):
                a1 = ArchD(); r1 = run_steps(cls(), a1, 5, n=20)
                a2 = ArchD(); r2 = run_steps(cls(), a2, 5, n=20)
                self.assertEqual(r1, r2)

    def test_predictor_error_declines_stationary(self):
        """§25 signature: with a fixed policy (stationary distribution),
        prediction error must decline on a learnable task."""
        env = ChangingRule()
        rng = new_rng(999)
        obs = env.reset(3)
        v = flatten_obs(obs)
        pred = LinearPredictor(len(v), env.action_space()["n"], 0.05, new_rng(1))
        errs = []
        done = False
        while not done:
            a = rng.randrange(env.action_space()["n"])
            obs2, r, done, _ = env.step(a)
            v2 = flatten_obs(obs2)
            errs.append(pred.update(v, a, v2, r))
            v = v2
        first = sum(errs[:20]) / 20
        last = sum(errs[-20:]) / 20
        self.assertLess(last, first, "predictor error did not decline")

    def test_snapshot_roundtrip(self):
        env, ag = ChangingRule(), ArchD()
        run_steps(env, ag, 5, n=15)
        snap = ag.snapshot()
        json.dumps(snap)
        ag2 = ArchD()
        ag2.restore(snap)
        self.assertEqual(ag.memory_stats()["store_size"],
                         ag2.memory_stats()["store_size"])

    def test_info_not_used(self):
        path = os.path.join(BASE, "..", "prototypes", "arch_d.py")
        with open(path) as fh:
            for i, line in enumerate(fh, 1):
                code = line.split("#")[0]
                self.assertNotRegex(code, r"\binfo\s*\[", f"arch_d.py:{i}")


if __name__ == "__main__":
    unittest.main()
