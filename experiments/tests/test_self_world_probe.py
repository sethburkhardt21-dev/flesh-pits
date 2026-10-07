"""Instrument tests for EXP-SW-01 (self/world distinction probe).

Validates the measurement chains (not the experimental verdicts):
- A: the specialist -> arbitrator chain CAN discriminate unequal stimuli
  (so A's null result is a finding, not a vacuous instrument);
- the reward adapter, policy streams, and verdict rules behave as specified;
- B's learning path moves weights and learns the self channel first.

Stdlib unittest. Deterministic (fixed seeds, no wall-clock).
"""

import os
import sys
import unittest

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "envs"))
sys.path.insert(0, os.path.join(EXP, "..", "prototypes", "architecture-a"))
sys.path.insert(0, os.path.join(EXP, "..", "prototypes", "architecture-b"))

from env_interface import derive_seed, new_rng, obs_to_vector  # noqa: E402
from self_world import SelfWorld  # noqa: E402
import exp_self_world_probe as probe  # noqa: E402
from attention import AttentionArbitrator  # noqa: E402


class TestAInstruments(unittest.TestCase):
    def test_change_specialists(self):
        specs = probe.make_change_specialists(["hand", "ball"])
        by_ch = dict(specs)
        s, _ = by_ch["hand"]({"hand": 0.5, "ball": 0.5}, 1)
        self.assertEqual(s, 0.0)  # first reading: no previous
        s, _ = by_ch["hand"]({"hand": 0.7, "ball": 0.5}, 2)
        self.assertAlmostEqual(s, 0.2)
        s, _ = by_ch["ball"]({"hand": 0.7, "ball": 0.5}, 2)
        self.assertEqual(s, 0.0)  # ball's first reading
        s, payload = by_ch["ball"]({"hand": 0.7, "ball": 0.9}, 3)
        self.assertAlmostEqual(s, 0.4)
        self.assertNotIn("action", payload)  # no efference copy in payload

    def test_chain_can_discriminate(self):
        # Instrument validity: after habituation to small constant deltas,
        # a large deviation on ONE channel bids clearly higher there.
        specs = probe.make_change_specialists(["hand", "ball"])
        arb = AttentionArbitrator(["hand", "ball"])
        o = {"hand": 0.5, "ball": 0.5}
        for i in range(1, 11):
            o = {"hand": 0.5 + 0.002 * i, "ball": 0.5 + 0.002 * i}
            stim = {c: fn(o, i)[0] for c, fn in specs}
            arb.arbitrate(stim)
        o = {"hand": 2.0, "ball": 0.5 + 0.002 * 10}  # big hand deviation
        stim = {c: fn(o, 11)[0] for c, fn in specs}
        d = arb.arbitrate(stim)
        self.assertGreater(d["competed_bids"]["hand"],
                           d["competed_bids"]["ball"] + 0.1)

    def test_chain_habituates_matched_stimuli(self):
        # The mechanism behind A's expected null: constant matched stimuli
        # habituate to the midpoint on BOTH channels.
        specs = probe.make_change_specialists(["hand", "ball"])
        arb = AttentionArbitrator(["hand", "ball"])
        o = {"hand": 0.5, "ball": 0.5}
        last = None
        for i in range(1, 31):
            o = {"hand": 0.5 + 0.2 * (i % 2), "ball": 0.5 + 0.2 * (i % 2)}
            stim = {c: fn(o, i)[0] for c, fn in specs}
            last = arb.arbitrate(stim)
        for c in ("hand", "ball"):
            self.assertAlmostEqual(last["competed_bids"][c], 0.5, delta=0.05)

    def test_reward_adapter(self):
        env = SelfWorld()
        env.reset(3)
        ad = probe.RewardAdapter(env, {"hand": 0, "ball": 1})
        r = ad.step("hand")
        self.assertIsInstance(r, float)
        self.assertEqual(r, 0.0)
        obs2, _, _, info = ad.last
        self.assertIn("hand", obs2)
        self.assertEqual(info["cause"]["hand"], "self")
        with self.assertRaises(KeyError):
            ad.step("nope")


class TestPolicyStreams(unittest.TestCase):
    def test_deterministic_and_paired(self):
        a1 = probe.policy_actions(75101, 0, 60)
        a2 = probe.policy_actions(75101, 0, 60)
        self.assertEqual(a1, a2)
        a3 = probe.policy_actions(75101, 1, 60)
        self.assertNotEqual(a1, a3)  # episodes differ
        self.assertTrue(all(a in (0, 1) for a in a1))

    def test_check_determinism(self):
        self.assertTrue(probe.check_determinism())


class TestVerdictRules(unittest.TestCase):
    def _b(self, di, df):
        return {"seed": 1, "n_heldout": 10, "intact_mean_e_hand": 0.0,
                "intact_mean_e_ball": 0.0, "frozen_mean_e_hand": 0.0,
                "frozen_mean_e_ball": 0.0, "D_intact": di, "D_frozen": df}

    def test_verdict_b_fires_on_consistent_gap(self):
        ps = [self._b(0.12, 0.01), self._b(0.11, -0.02),
              self._b(0.13, 0.03), self._b(0.10, 0.00)]
        v = probe.verdict_b(ps)
        self.assertTrue(v["rule_fired"])
        self.assertIn("PASS", v["verdict"])

    def test_verdict_b_null_without_gap(self):
        ps = [self._b(0.01, 0.01), self._b(-0.02, 0.00),
              self._b(0.02, 0.02), self._b(0.00, -0.01)]
        v = probe.verdict_b(ps)
        self.assertFalse(v["rule_fired"])

    def _a(self, db, di):
        return {"seed": 1, "n_ticks": 10, "mean_bid_hand": 0.0,
                "mean_bid_ball": 0.0, "Delta_bid": db, "ign_rate_hand": 0.0,
                "ign_rate_ball": 0.0, "Delta_ign": di,
                "mean_strength_hand": 0.0, "mean_strength_ball": 0.0,
                "gain_hand": 1.0, "gain_ball": 1.0,
                "Delta_gain_diagnostic": 0.0}

    def test_verdict_a_null_on_matched(self):
        ps = [self._a(0.001, 0.0), self._a(-0.001, 0.0),
              self._a(0.002, 0.0), self._a(0.001, 0.0)]
        v = probe.verdict_a(ps)
        self.assertFalse(v["rule_fired"])
        self.assertIn("NO DISTINCTION", v["verdict"])

    def test_verdict_a_fires_on_real_gap(self):
        ps = [self._a(0.10, 0.0), self._a(0.12, 0.0),
              self._a(0.09, 0.0), self._a(0.11, 0.0)]
        v = probe.verdict_a(ps)
        self.assertTrue(v["rule_fired"])


class TestBLearningPath(unittest.TestCase):
    def test_learn_transition_moves_weights(self):
        from agent import ArchB
        env = SelfWorld()
        space = env.observation_space()
        ag = ArchB(space, 2, env_name="self_world", affect="none", seed=75001)
        w0 = [row[:] for row in ag.model.W]
        obs = env.reset(derive_seed(75101, 0, "env"))
        for a in probe.policy_actions(75101, 0, 60):
            obs2, r, _, _ = env.step(a)
            ag.learn_transition(obs_to_vector(obs, space), a,
                                obs_to_vector(obs2, space), r)
            obs = obs2
        moved = sum(abs(x - y) for r0, r1 in zip(w0, ag.model.W)
                    for x, y in zip(r0, r1))
        self.assertGreater(moved, 0.0)

    def test_frozen_arm_learns_nothing(self):
        from agent import ArchB
        env = SelfWorld()
        space = env.observation_space()
        ag = ArchB(space, 2, env_name="self_world", affect="none",
                   seed=75001, frozen=True)
        w0 = [row[:] for row in ag.model.W]
        obs = env.reset(derive_seed(75101, 0, "env"))
        for a in probe.policy_actions(75101, 0, 60):
            obs2, r, _, _ = env.step(a)
            ag.learn_transition(obs_to_vector(obs, space), a,
                                obs_to_vector(obs2, space), r)
            obs = obs2
        self.assertEqual(w0, ag.model.W)

    def test_self_channel_learned_better_than_world(self):
        # Behavioral: after modest training the deterministic self channel
        # is predicted better than the exogenous world channel.
        from agent import ArchB
        env = SelfWorld()
        space = env.observation_space()
        ag = ArchB(space, 2, env_name="self_world", affect="none", seed=75001)
        for ep in range(5):
            obs = env.reset(derive_seed(75101, ep, "env"))
            for a in probe.policy_actions(75101, ep, 60):
                obs2, r, _, _ = env.step(a)
                ag.learn_transition(obs_to_vector(obs, space), a,
                                    obs_to_vector(obs2, space), r)
                obs = obs2
        eh, eb = [], []
        obs = env.reset(derive_seed(75101, 900, "env"))
        for a in probe.policy_actions(75101, 900, 60):
            obs2, _, _, _ = env.step(a)
            v, v2 = obs_to_vector(obs, space), obs_to_vector(obs2, space)
            xhat = ag.model.predict_next(v, a, ag.context_key(v))
            eh.append(abs(v2[0] - xhat[0]))
            eb.append(abs(v2[1] - xhat[1]))
            obs = obs2
        self.assertLess(sum(eh) / len(eh), sum(eb) / len(eb))


if __name__ == "__main__":
    unittest.main()
