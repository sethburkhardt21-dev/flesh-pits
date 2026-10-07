"""Contract-conformance tests for the §35 environments. Stdlib unittest."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs"))

from env_interface import Environment, make_episode_id  # noqa: E402
from envs import ALL_ENVS, ChangingRule  # noqa: E402


def rollout(cls, seed, n=40):
    env = cls()
    obs = env.reset(seed)
    init_hash = env.state_hash()
    traj = []
    done = False
    i = 0
    while not done and i < n:
        obs, r, done, info = env.step(i % env.action_space()["n"])
        traj.append((round(r, 6), done))
        i += 1
    return obs, init_hash, traj, env.state_hash(), info


class TestEnvs(unittest.TestCase):
    def test_determinism(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                a = rollout(cls, 42)
                b = rollout(cls, 42)
                self.assertEqual(a, b, f"{cls.NAME} not deterministic")

    def test_seed_sensitivity(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                _, h1, _, _, _ = rollout(cls, 42)
                _, h2, _, _, _ = rollout(cls, 43)
                self.assertNotEqual(h1, h2, f"{cls.NAME} ignores seed")

    def test_invalid_action_raises(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                env.reset(1)
                with self.assertRaises(ValueError):
                    env.step(999)
                with self.assertRaises(ValueError):
                    env.step(-1)

    def test_obs_matches_space(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                obs = env.reset(7)
                env.validate_obs(obs, env.observation_space())  # raises if bad
                done = False
                i = 0
                while not done and i < 60:
                    obs, _, done, _ = env.step(i % env.action_space()["n"])
                    env.validate_obs(obs, env.observation_space())
                    i += 1

    def test_state_hash_changes(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                env.reset(7)
                h0 = env.state_hash()
                env.step(0)
                self.assertNotEqual(h0, env.state_hash())

    def test_episode_id_format(self):
        eid = make_episode_id("grid_world", "1.0.0", 42, 3)
        self.assertEqual(eid, "grid_world:v1.0.0:seed=42:run=3")

    def test_changing_rule_phase_schedule(self):
        env = ChangingRule()
        rules = []
        for ep in range(12):
            env.reset(1000 + ep)
            rules.append(tuple(env._rule))
        # fixed within a phase, flips between phases, schedule run-stable
        self.assertEqual(rules[0], rules[4])
        self.assertEqual(rules[5], rules[9])
        self.assertNotEqual(rules[0], rules[5])
        env2 = ChangingRule()
        rules2 = []
        for ep in range(12):
            env2.reset(555 + ep)
            rules2.append(tuple(env2._rule))
        self.assertEqual(rules, rules2)

    def test_all_envs_expose_contract(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                self.assertTrue(issubclass(cls, Environment))
                self.assertTrue(cls.NAME and cls.VERSION)
                env = cls()
                self.assertEqual(env.action_space()["type"], "discrete")


if __name__ == "__main__":
    unittest.main()
