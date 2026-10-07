"""Tests for the experiment harness. Stdlib unittest."""

import json
import os
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import harness  # noqa: E402
from env_interface import derive_seed  # noqa: E402


class TestHarness(unittest.TestCase):
    def test_run_experiment_shape(self):
        r = harness.run_experiment("grid_world", "stateless", 4, 7, "EXP-TEST")
        self.assertEqual(r["experiment_id"], "EXP-TEST")
        self.assertEqual(len(r["episodes"]), 4)
        self.assertIn("config_hash", r)
        s = r["summary"]
        for k in ("mean_return", "stdev_return", "min_return", "max_return",
                  "mean_steps", "n_episodes"):
            self.assertIn(k, s)
        ep = r["episodes"][0]
        for k in ("episode_id", "seed", "init_hash", "final_hash", "return",
                  "steps", "done", "truncated"):
            self.assertIn(k, ep)
        self.assertTrue(ep["episode_id"].startswith("grid_world:v"))

    def test_reproducible(self):
        a = harness.run_experiment("changing_rule", "arch_d", 3, 11, "X")
        b = harness.run_experiment("changing_rule", "arch_d", 3, 11, "X")
        self.assertEqual(a["config_hash"], b["config_hash"])
        self.assertEqual([e["return"] for e in a["episodes"]],
                         [e["return"] for e in b["episodes"]])
        self.assertEqual([e["final_hash"] for e in a["episodes"]],
                         [e["final_hash"] for e in b["episodes"]])

    def test_seed_derivation_independent_streams(self):
        self.assertNotEqual(derive_seed(7, 0, "env"), derive_seed(7, 0, "agent"))
        self.assertNotEqual(derive_seed(7, 0, "env"), derive_seed(7, 1, "env"))

    def test_receipt_schema(self):
        r = harness.run_experiment("delayed_reward", "symbolic", 2, 13, "EXP-R")
        with tempfile.TemporaryDirectory() as td:
            path = harness.write_receipt(r, td, hypothesis="h", null="n",
                                         preregistered_metric="mean_return",
                                         baseline="b", conditions="c",
                                         interpretation="i", limitations="l")
            with open(path) as fh:
                rec = json.load(fh)
        for k in ("experiment_id", "hypothesis", "null_hypothesis",
                  "preregistered_metric", "baseline", "conditions",
                  "config", "config_hash", "primary_seed", "started_utc",
                  "written_utc", "episodes", "summary",
                  "interpretation", "limitations"):
            self.assertIn(k, rec, f"receipt missing {k}")

    def test_learning_accumulates_across_episodes(self):
        """Same agent object across episodes: no_metacognition's Q-table
        must differ after experience (persistence, not reset-wiped)."""
        from baselines import NoMetacognitionAgent
        from envs import GridWorld
        env, agent = GridWorld(), NoMetacognitionAgent()
        q0 = None
        for run in range(3):
            obs = env.reset(derive_seed(5, run, "env"))
            agent.reset(derive_seed(5, run, "agent"), env.action_space())
            if run == 0:
                q0 = dict(agent._Q)
            done = False
            while not done:
                a = agent.act(obs)
                obs2, r, done, info = env.step(a)
                agent.update(obs, a, r, done, info)
                obs = obs2
        self.assertGreater(len(agent._Q), len(q0),
                           "Q-table did not grow: cross-episode learning broken")


if __name__ == "__main__":
    unittest.main()
