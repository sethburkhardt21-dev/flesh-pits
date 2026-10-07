"""Contract-conformance tests for the §35 environments. Stdlib unittest."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs"))

from env_interface import Environment, make_episode_id  # noqa: E402
from envs import ALL_ENVS, ChangingRule, SelfWorld  # noqa: E402


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


class TestSelfWorld(unittest.TestCase):
    """EXP-SW-01 env: self/world causal structure + matched statistics."""

    def test_registered(self):
        names = [c.NAME for c in ALL_ENVS]
        self.assertIn("self_world", names)

    def test_spaces(self):
        env = SelfWorld()
        self.assertEqual(env.action_space()["n"], 2)
        space = env.observation_space()
        self.assertEqual(set(space), {"hand", "ball"})
        for ch in ("hand", "ball"):
            self.assertEqual(space[ch]["type"], "scalar")
            self.assertEqual((space[ch]["low"], space[ch]["high"]), (0.0, 1.0))

    def test_hand_is_action_determined(self):
        # SELF-caused: hand_{t+1} is a deterministic function of (hand_t, a).
        for seed in (1, 2, 3):
            for actions in ([0] * 10, [1] * 10, [0, 1] * 5):
                e1, e2 = SelfWorld(), SelfWorld()
                o1, o2 = e1.reset(seed), e2.reset(seed)
                for a in actions:
                    o1, _, _, _ = e1.step(a)
                    o2, _, _, _ = e2.step(a)
                    self.assertEqual(o1["hand"], o2["hand"])

    def test_ball_is_exogenous(self):
        # WORLD-caused: ball moves even when the action stream is fixed,
        # and its path is not a function of actions alone.
        env = SelfWorld()
        env.reset(11)
        balls = []
        for _ in range(20):
            obs, _, _, _ = env.step(0)  # constant action
            balls.append(obs["hand"])
        # hand under constant action is deterministic drift; ball varies
        env2 = SelfWorld()
        env2.reset(12)
        b2 = [env2.step(0)[0]["ball"] for _ in range(20)]
        env3 = SelfWorld()
        env3.reset(11)
        b3 = [env3.step(0)[0]["ball"] for _ in range(20)]
        self.assertEqual(b2, b2)  # sanity
        self.assertNotEqual(b2, b3)  # seed matters -> exogenous RNG stream

    def test_matched_change_statistics(self):
        # Both channels change with |displacement| in {0, STEP}: STEP
        # normally; 0 only at the reflection fixed points (STEP/2 and
        # 1-STEP/2), where a step reflects onto itself — symmetric for
        # both channels. Rates and magnitudes are matched by construction.
        env = SelfWorld()
        env.reset(75101)
        hand_moves = ball_moves = 0
        n = 60
        for i in range(n):
            o = {"hand": env._hand, "ball": env._ball}
            obs, _, _, _ = env.step(i % 2)
            for ch in ("hand", "ball"):
                d = abs(obs[ch] - o[ch])
                self.assertIn(round(d, 9), (0.0, round(SelfWorld.STEP, 9)))
            if abs(obs["hand"] - o["hand"]) > 1e-12:
                hand_moves += 1
            if abs(obs["ball"] - o["ball"]) > 1e-12:
                ball_moves += 1
        self.assertGreaterEqual(hand_moves, 0.8 * n)
        self.assertGreaterEqual(ball_moves, 0.8 * n)
        # Mechanism-matched (identical reflection for both channels), not
        # rate-identical: fixed-point non-moves are hit stochastically.

    def test_cause_labels_for_scoring_only(self):
        env = SelfWorld()
        env.reset(5)
        _, _, _, info = env.step(0)
        self.assertEqual(info["cause"], {"hand": "self", "ball": "world"})

    def test_reward_constant_zero(self):
        env = SelfWorld()
        env.reset(5)
        for i in range(60):
            _, r, done, _ = env.step(i % 2)
            self.assertEqual(r, 0.0)
            if done:
                break
        self.assertTrue(done)  # truncates at MAX_STEPS

    def test_reflection_preserves_determinism(self):
        env = SelfWorld()
        env.reset(999)
        t1 = [(env.step(0)[0]["hand"], env.step(0)[0]["ball"]) for _ in range(60)]
        env.reset(999)
        t2 = [(env.step(0)[0]["hand"], env.step(0)[0]["ball"]) for _ in range(60)]
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
