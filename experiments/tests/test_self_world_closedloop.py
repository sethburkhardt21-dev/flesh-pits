"""Instrument tests for EXP-SW-02-B (closed-loop self/world extension).

Validates the measurement chains (not the experimental verdicts):
- self_world_cl reward semantics (additive; canonical self_world untouched);
- the greedy planner prefers the action moving the hand toward target when
  the model predicts correctly, and ties keep the last action;
- the shuffle-label stream differs from the policy stream (lesion is real);
- the verdict rule fires exactly per the preregistered decision rule.

Stdlib unittest. Deterministic (fixed seeds, no wall-clock).
"""

import os
import sys
import unittest

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "envs"))
sys.path.insert(0, os.path.join(EXP, "..", "prototypes", "architecture-b"))

from env_interface import derive_seed, new_rng  # noqa: E402
from self_world import SelfWorld  # noqa: E402
from self_world_cl import SelfWorldCL  # noqa: E402
import exp_self_world_closedloop as cl  # noqa: E402


class TestSelfWorldCL(unittest.TestCase):
    def test_reward_semantics(self):
        env = SelfWorldCL()
        env.reset(7)
        # action 1 moves hand 0.5 -> 0.7: reward = -(0.7-0.5)^2
        obs, r, done, info = env.step(1)
        self.assertAlmostEqual(obs["hand"], 0.7)
        self.assertAlmostEqual(r, -(0.2 ** 2))
        self.assertFalse(done)
        self.assertEqual(info["cause"], {"hand": "self", "ball": "world"})

    def test_canonical_env_untouched(self):
        self.assertEqual(SelfWorld.NAME, "self_world")
        self.assertEqual(SelfWorld.VERSION, "1.0.0")
        env = SelfWorld()
        env.reset(7)
        _, r, _, _ = env.step(1)
        self.assertEqual(r, 0.0)  # canonical env still zero-reward

    def test_distinct_identity(self):
        self.assertEqual(SelfWorldCL.NAME, "self_world_cl")
        self.assertEqual(SelfWorldCL.VERSION, "1.0.0")
        self.assertEqual(SelfWorldCL.TARGET, 0.5)
        a, b = SelfWorldCL(), SelfWorld()
        a.reset(7)
        b.reset(7)
        self.assertNotEqual(a.state_hash(), b.state_hash())

    def test_envs_registered(self):
        from envs import ALL_ENVS
        names = [c.NAME for c in ALL_ENVS]
        self.assertIn("self_world", names)
        self.assertIn("self_world_cl", names)


class StubModel:
    """Perfect hand predictor: hand_{t+1} = reflect(hand +/- 0.2)."""

    def predict_next(self, v, a, ctx):
        d = -0.2 if a == 0 else 0.2
        h = v[0] + d
        h = -h if h < 0.0 else (2.0 - h if h > 1.0 else h)
        return [h, v[1]]


class StubAgent:
    def __init__(self):
        self.model = StubModel()

    def context_key(self, v):
        return ()


class TestGreedyPlanner(unittest.TestCase):
    def test_moves_hand_toward_target(self):
        agent = StubAgent()
        space = SelfWorldCL().observation_space()
        # hand 0.7 -> action 0 (left) moves toward 0.5
        self.assertEqual(cl.greedy_action(agent, space,
                                          {"hand": 0.7, "ball": 0.5}, 1), 0)
        # hand 0.3 -> action 1 (right)
        self.assertEqual(cl.greedy_action(agent, space,
                                          {"hand": 0.3, "ball": 0.9}, 0), 1)

    def test_tie_keeps_last_action(self):
        # exact float ties are measure-zero; test the rule with a stub
        # whose costs are bit-identical for both actions.
        class TieModel:
            def predict_next(self, v, a, ctx):
                return [v[0], v[1]]  # cost identical for a in {0,1}

        class TieAgent:
            def __init__(self):
                self.model = TieModel()

            def context_key(self, v):
                return ()

        agent = TieAgent()
        space = SelfWorldCL().observation_space()
        self.assertEqual(cl.greedy_action(agent, space,
                                          {"hand": 0.5, "ball": 0.5}, 1), 1)
        self.assertEqual(cl.greedy_action(agent, space,
                                          {"hand": 0.5, "ball": 0.5}, 0), 0)


class TestLesionStreams(unittest.TestCase):
    def test_shuffle_stream_differs_from_policy(self):
        p = cl.policy_actions(75201, 0, 60)
        s = cl.shuffle_actions(75201, 0, 60)
        self.assertNotEqual(p, s)  # lesion must actually relabel
        # both are valid binary streams
        self.assertTrue(all(a in (0, 1) for a in p + s))
        # shuffle stream is deterministic
        self.assertEqual(s, cl.shuffle_actions(75201, 0, 60))

    def test_determinism_gate(self):
        self.assertTrue(cl.check_determinism())


class TestVerdictRule(unittest.TestCase):
    def _per_seed(self, gaps):
        return [{"R_gap_attrib": g, "R_gap_learn": g,
                 "return_intact": 0.0, "return_shuffle": 0.0,
                 "return_frozen": 0.0, "online_D_intact": 0.0,
                 "g0c_intact": True} for g in gaps]

    def test_pass_rule(self):
        res = cl.verdict(self._per_seed([19.0, 17.5, 20.1, 18.2]))
        self.assertEqual(res["verdict"], "ATTRIBUTION ADVANTAGE (PASS)")
        self.assertTrue(res["rule_fired"])

    def test_fail_on_weak_mean(self):
        res = cl.verdict(self._per_seed([0.5, 0.4, 0.6, 0.5]))
        self.assertIn("NO ATTRIBUTION ADVANTAGE", res["verdict"])
        self.assertFalse(res["rule_fired"])

    def test_fail_on_inconsistent_sign(self):
        res = cl.verdict(self._per_seed([5.0, 4.0, -1.0, -0.5]))
        self.assertFalse(res["rule_fired"])


if __name__ == "__main__":
    unittest.main()
