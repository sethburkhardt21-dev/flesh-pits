"""Unit tests for attention_ecr_repro.py (ReturnConditionedEpisodicArbitrator).

Independent tests written from the EXP-FP-0080 preregistered rule summary.
No tick/env dependency: the (winner, reward) stream is fed directly.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)

from attention_ecr_repro import ReturnConditionedEpisodicArbitrator

CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
BRANCHES = {"branch_a", "branch_b"}


def make(**kw):
    kw.setdefault("branch_channels", BRANCHES)
    kw.setdefault("max_steps", 15)
    return ReturnConditionedEpisodicArbitrator(CHANNELS, **kw)


class TestConstruction(unittest.TestCase):
    def test_requires_priors_to_update(self):
        a = ReturnConditionedEpisodicArbitrator(CHANNELS)  # no priors
        a._pending_winner = "forward"
        with self.assertRaises(ValueError):
            a.update_gains({"forward": 0.02})

    def test_rejects_non_subset_branch_channels(self):
        with self.assertRaises(ValueError):
            ReturnConditionedEpisodicArbitrator(
                CHANNELS, branch_channels={"nope"}, max_steps=15)

    def test_rejects_nonpositive_max_steps(self):
        with self.assertRaises(ValueError):
            ReturnConditionedEpisodicArbitrator(
                CHANNELS, branch_channels=BRANCHES, max_steps=0)

    def test_frozen_gains_never_move(self):
        a = make(frozen=True)
        a._pending_winner = "forward"
        for _ in range(3):
            a.update_gains({"forward": 1.02})  # even +1.0 must not move
        self.assertEqual(a.gains, {c: 1.0 for c in CHANNELS})


class TestSuccessCredit(unittest.TestCase):
    def _episode(self, winners, final_reward=1.02):
        a = make()
        for i, w in enumerate(winners):
            a._pending_winner = w
            r = final_reward if i == len(winners) - 1 else (
                0.02 if w == "forward" else 0.0)
            a.update_gains({w: r})
        return a

    def test_success_sets_branch_choice_and_progress_to_cap(self):
        a = self._episode(["branch_a"] + ["forward"] * 9)
        self.assertEqual(a.gains["branch_a"], a.gain_cap)
        self.assertEqual(a.gains["forward"], a.gain_cap)
        # Unused branch never credited by the success.
        self.assertEqual(a.gains["branch_b"], 1.0)

    def test_first_branch_win_is_credited(self):
        a = self._episode(["forward", "branch_b"] + ["forward"] * 9)
        self.assertEqual(a.gains["branch_b"], a.gain_cap)

    def test_ledger_clears_on_success(self):
        a = self._episode(["branch_a"] + ["forward"] * 9)
        self.assertEqual(a._ledger.winners, [])
        self.assertEqual(a._ledger.ticks, 0)

    def test_truncation_clears_without_cap_sets(self):
        a = make()
        for _ in range(15):
            a._pending_winner = "stay"
            a.update_gains({"stay": 0.0})
        self.assertEqual(a._ledger.winners, [])
        self.assertEqual(a._ledger.ticks, 0)
        self.assertTrue(all(v < a.gain_cap for v in a.gains.values()))


class TestShaping(unittest.TestCase):
    def test_shaping_punishes_useless_protects_branches_boosts_winner(self):
        a = make()
        a._pending_winner = "forward"
        a.update_gains({"forward": 0.02})
        self.assertLess(a.gains["stay"], 1.0)
        self.assertEqual(a.gains["branch_a"], 1.0)
        self.assertEqual(a.gains["branch_b"], 1.0)
        self.assertGreater(a.gains["forward"], 1.0)

    def test_progress_channel_is_episode_forward(self):
        a = make()
        for _ in range(5):
            a._pending_winner = "forward"
            a.update_gains({"forward": 0.02})
        self.assertEqual(a._ledger.progress_channel(a.channels), "forward")

    def test_punish_floor_holds(self):
        a = make()
        for _ in range(200):  # far past the floor
            a._pending_winner = "forward"
            a.update_gains({"forward": 0.02})
        self.assertGreaterEqual(a.gains["stay"], 0.01)


class TestPreBranchNoop(unittest.TestCase):
    def test_demote_when_above_branch_ceiling(self):
        a = make()
        a.gains["forward"] = 1.5
        a._pending_winner = "forward"
        a.update_gains({"forward": 0.0})
        self.assertLess(a.gains["forward"], 1.5)
        self.assertGreaterEqual(a.gains["forward"], 1.0)

    def test_no_demote_at_or_below_branch_ceiling(self):
        a = make()  # forward == branches == 1.0
        a._pending_winner = "forward"
        a.update_gains({"forward": 0.0})
        self.assertEqual(a.gains["forward"], 1.0)

    def test_branch_winners_never_demoted_prebranch(self):
        a = make()
        a.gains["branch_a"] = 1.5
        a._pending_winner = "branch_a"
        a.update_gains({"branch_a": 0.0})
        self.assertEqual(a.gains["branch_a"], 1.5)

    def test_no_demote_post_branch(self):
        a = make()
        a.gains["forward"] = 1.5
        a._pending_winner = "branch_a"
        a.update_gains({"branch_a": 0.0})  # branch chosen
        a._pending_winner = "forward"
        a.update_gains({"forward": 0.0})  # post-branch r=0: no change
        self.assertEqual(a.gains["forward"], 1.5)


class TestDeterminism(unittest.TestCase):
    def test_identical_streams_identical_gains(self):
        def run():
            a = make()
            for w, r in [("branch_a", 0.0), ("forward", 0.02),
                         ("forward", 0.02), ("stay", 0.0),
                         ("forward", 1.02)]:
                a._pending_winner = w
                a.update_gains({w: r})
            return dict(a.gains)
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
