"""Unit tests for attention_ecr.py (EpisodicContrastiveArbitrator).

Verifies the ECR gain-update rule in isolation (no tick/env).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)

from attention_ecr import EpisodicContrastiveArbitrator

CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
BRANCHES = {"branch_a", "branch_b"}


def make(**kw):
    kw.setdefault("branch_channels", BRANCHES)
    kw.setdefault("max_steps", 15)
    return EpisodicContrastiveArbitrator(CHANNELS, **kw)


class TestECRConstruction(unittest.TestCase):
    def test_requires_branch_channels_for_update(self):
        a = EpisodicContrastiveArbitrator(CHANNELS)  # no priors
        a._last_winner = "forward"
        with self.assertRaises(ValueError):
            a.update_gains({"forward": 0.02})

    def test_rejects_bad_branch_channels(self):
        with self.assertRaises(ValueError):
            EpisodicContrastiveArbitrator(
                CHANNELS, branch_channels={"nope"}, max_steps=15)

    def test_rejects_bad_max_steps(self):
        with self.assertRaises(ValueError):
            EpisodicContrastiveArbitrator(
                CHANNELS, branch_channels=BRANCHES, max_steps=0)

    def test_frozen_no_update(self):
        a = make(frozen=True)
        a._last_winner = "forward"
        # Even a +1.0 must not move frozen gains.
        for _ in range(3):
            a.update_gains({"forward": 1.02})
        self.assertEqual(a.gains,
                         {c: 1.0 for c in CHANNELS})


class TestECRSuccess(unittest.TestCase):
    def test_success_locks_branch_and_progress_to_cap(self):
        a = make()
        # Simulate a successful episode: branch_a at t=0, then forwards.
        winners = ["branch_a"] + ["forward"] * 9
        for i, w in enumerate(winners):
            a._last_winner = w
            r = 1.02 if i == len(winners) - 1 else (0.02 if w == "forward"
                                                    else 0.0)
            a.update_gains({w: r})
        self.assertEqual(a.gains["branch_a"], a.gain_cap)
        self.assertEqual(a.gains["forward"], a.gain_cap)
        # branch_b and stay untouched by the success credit.
        self.assertEqual(a.gains["branch_b"], 1.0)

    def test_success_credits_first_branch_win(self):
        a = make()
        # forward no-op at t=0, then branch_b, then forwards.
        winners = ["forward", "branch_b"] + ["forward"] * 9
        for i, w in enumerate(winners):
            a._last_winner = w
            r = 1.02 if i == len(winners) - 1 else (0.02 if w == "forward"
                                                    else 0.0)
            a.update_gains({w: r})
        # branch_b (the actual branch choice), not the t=0 forward no-op.
        self.assertEqual(a.gains["branch_b"], a.gain_cap)

    def test_episode_resets_after_success(self):
        a = make()
        a._last_winner = "branch_a"
        a.update_gains({"branch_a": 0.0})
        a._last_winner = "forward"
        a.update_gains({"forward": 1.02})
        self.assertEqual(a._ep_winners, [])
        self.assertEqual(a._ep_ticks, 0)

    def test_truncation_resets_without_credit(self):
        a = make()
        for _ in range(15):
            a._last_winner = "stay"
            a.update_gains({"stay": 0.0})
        # 15 ticks, no success -> truncation reset, no cap sets.
        self.assertEqual(a._ep_winners, [])
        self.assertEqual(a._ep_ticks, 0)
        self.assertTrue(all(v < a.gain_cap for v in a.gains.values()))


class TestECRShaping(unittest.TestCase):
    def test_shaping_demotes_useless_channels(self):
        a = make()
        a._last_winner = "forward"
        a.update_gains({"forward": 0.02})
        # stay demoted; branches protected; forward boosted slightly.
        self.assertLess(a.gains["stay"], 1.0)
        self.assertEqual(a.gains["branch_a"], 1.0)
        self.assertEqual(a.gains["branch_b"], 1.0)
        self.assertGreater(a.gains["forward"], 1.0)

    def test_shaping_counts_identify_progress(self):
        a = make()
        for _ in range(5):
            a._last_winner = "forward"
            a.update_gains({"forward": 0.02})
        self.assertEqual(a._progress_channel(), "forward")


class TestECRPreBranch(unittest.TestCase):
    def test_prebranch_demotes_forward_above_branches(self):
        a = make()
        a.gains["forward"] = 1.5  # artificially high
        a._last_winner = "forward"
        a.update_gains({"forward": 0.0})  # pre-branch no-op
        self.assertLess(a.gains["forward"], 1.5)
        self.assertGreaterEqual(a.gains["forward"], 1.0)  # not below branches

    def test_prebranch_does_not_demote_below_branches(self):
        a = make()  # forward at 1.0 == branches
        a._last_winner = "forward"
        a.update_gains({"forward": 0.0})
        # forward not above branches -> untouched.
        self.assertEqual(a.gains["forward"], 1.0)

    def test_prebranch_never_demotes_branch(self):
        a = make()
        a.gains["branch_a"] = 1.5
        a._last_winner = "branch_a"
        a.update_gains({"branch_a": 0.0})
        self.assertEqual(a.gains["branch_a"], 1.5)

    def test_postbranch_no_demotion(self):
        a = make()
        a.gains["forward"] = 1.5
        # branch chosen first, then forward no-op (post-branch).
        a._last_winner = "branch_a"
        a.update_gains({"branch_a": 0.0})
        a._last_winner = "forward"
        a.update_gains({"forward": 0.0})
        # post-branch r=0: no change.
        self.assertEqual(a.gains["forward"], 1.5)


class TestECRDeterminism(unittest.TestCase):
    def test_deterministic(self):
        def run():
            a = make()
            seq = [("branch_a", 0.0), ("forward", 0.02), ("forward", 0.02),
                   ("stay", 0.0), ("forward", 1.02)]
            for w, r in seq:
                a._last_winner = w
                a.update_gains({w: r})
            return dict(a.gains)
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
