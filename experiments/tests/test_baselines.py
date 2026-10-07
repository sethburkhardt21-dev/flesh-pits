"""Tests for the §39 baseline agents. Stdlib unittest."""

import json
import os
import re
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "envs"))
sys.path.insert(0, os.path.join(BASE, "baselines"))

from envs import ALL_ENVS, GridWorld  # noqa: E402
from baselines import ALL_BASELINES  # noqa: E402


def run_steps(env, agent, seed, n=30):
    obs = env.reset(seed)
    agent.reset(seed ^ 0x9E3779B9, env.action_space())
    seq = []
    done = False
    i = 0
    while not done and i < n:
        a = agent.act(obs)
        seq.append(a)
        obs2, r, done, info = env.step(a)
        agent.update(obs, a, r, done, info)
        obs = obs2
        i += 1
    return seq, obs


class TestBaselines(unittest.TestCase):
    def test_all_run_all_envs_valid_actions(self):
        for ecls in ALL_ENVS:
            for acls in ALL_BASELINES:
                with self.subTest(env=ecls.NAME, agent=acls.NAME):
                    env, agent = ecls(), acls()
                    seq, _ = run_steps(env, agent, 7, n=25)
                    self.assertTrue(len(seq) > 0)
                    for a in seq:
                        self.assertTrue(0 <= a < env.action_space()["n"])

    def test_determinism(self):
        for acls in ALL_BASELINES:
            with self.subTest(agent=acls.NAME):
                s1, _ = run_steps(GridWorld(), acls(), 99, n=25)
                s2, _ = run_steps(GridWorld(), acls(), 99, n=25)
                self.assertEqual(s1, s2, f"{acls.NAME} nondeterministic")

    def test_snapshot_json_and_restore(self):
        for acls in ALL_BASELINES:
            with self.subTest(agent=acls.NAME):
                env, agent = GridWorld(), acls()
                seq, obs = run_steps(env, agent, 5, n=10)
                snap = agent.snapshot()
                json.dumps(snap)  # must be JSON-serializable
                a_before = agent.act(obs)
                agent2 = acls()
                agent2.reset(5 ^ 0x9E3779B9, env.action_space())
                agent2.restore(snap)
                self.assertEqual(a_before, agent2.act(obs),
                                 f"{acls.NAME} restore mismatch")

    def test_info_never_drives_decisions(self):
        """Static audit: no baseline may index into `info` (contract rule)."""
        for path in [os.path.join(BASE, "baselines", f)
                     for f in os.listdir(os.path.join(BASE, "baselines"))
                     if f.endswith(".py")]:
            with open(path) as fh:
                for i, line in enumerate(fh, 1):
                    code = line.split("#")[0]
                    self.assertNotRegex(code, r"\binfo\s*\[",
                                        f"{path}:{i} indexes info")
                    self.assertNotRegex(code, r"\binfo\s*\.\s*get\s*\(",
                                        f"{path}:{i} reads info")


if __name__ == "__main__":
    unittest.main()
