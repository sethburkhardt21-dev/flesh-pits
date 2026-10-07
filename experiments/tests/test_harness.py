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
import id_registry  # noqa: E402


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


def _fake_result(exp_id):
    """Minimal harness result body for receipt tests (no env run needed)."""
    return {
        "experiment_id": exp_id,
        "config": {"env": "grid_world", "agent": "stateless"},
        "config_hash": "deadbeef",
        "primary_seed": 1,
        "started_utc": "2026-10-07T00:00:00+00:00",
        "episodes": [],
        "summary": {"n_episodes": 0},
    }


class TestWriteReceiptFailClosed(unittest.TestCase):
    """2026-10-07 EXP-FP-0050 collision: write_receipt must never silently
    overwrite an existing receipt. Runs on a FRESH temp receipts dir; no
    existing receipt is touched."""

    def test_existing_path_without_supersede_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = harness.write_receipt(_fake_result("EXP-DUP"), td)
            with open(p1) as f:
                before = json.load(f)
            with self.assertRaises(FileExistsError):
                harness.write_receipt(_fake_result("EXP-DUP"), td)
            # the original receipt is untouched: byte-identical, no
            # supersede fields
            with open(p1) as f:
                after = json.load(f)
            self.assertEqual(before, after)
            self.assertNotIn("supersede_reason", after)
            self.assertNotIn("superseded_previous_hash", after)

    def test_existing_path_empty_supersede_still_raises(self):
        with tempfile.TemporaryDirectory() as td:
            harness.write_receipt(_fake_result("EXP-DUP2"), td)
            with self.assertRaises(FileExistsError):
                harness.write_receipt(_fake_result("EXP-DUP2"), td,
                                      supersede="")
            with self.assertRaises(FileExistsError):
                harness.write_receipt(_fake_result("EXP-DUP2"), td,
                                      supersede=None)

    def test_supersede_records_previous_hash_and_reason(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = harness.write_receipt(_fake_result("EXP-SUP"), td)
            with open(p1) as f:
                old_hash = json.load(f)["receipt_hash"]
            r2 = _fake_result("EXP-SUP")
            r2["summary"] = {"n_episodes": 1, "note": "re-run"}
            reason = "re-run after crash truncated the first write"
            p2 = harness.write_receipt(r2, td, supersede=reason)
            self.assertEqual(p1, p2)
            with open(p2) as f:
                rec = json.load(f)
            self.assertEqual(rec["superseded_previous_hash"], old_hash)
            self.assertEqual(rec["supersede_reason"], reason)
            self.assertEqual(rec["summary"]["note"], "re-run")
            # the superseded receipt still verifies in the chain
            ok, problems = harness.verify_chain(td)
            self.assertTrue(ok, problems)

    def test_supersede_hashless_old_receipt_records_none(self):
        with tempfile.TemporaryDirectory() as td:
            legacy = os.path.join(td, "EXP-LEG.json")
            with open(legacy, "w") as f:
                json.dump({"experiment_id": "EXP-LEG"}, f)
            rec = json.load(open(
                harness.write_receipt(_fake_result("EXP-LEG"), td,
                                      supersede="adopt legacy receipt")))
            self.assertIsNone(rec["superseded_previous_hash"])
            self.assertEqual(rec["supersede_reason"], "adopt legacy receipt")


class TestIDRegistry(unittest.TestCase):
    """Per-lane ID allocation registry (2026-10-07 EXP-FP-0050 systemic
    fix). All tests run against registries in FRESH temp dirs; the real
    experiments/ID_REGISTRY is never touched."""

    def test_seed_records_preallocations_and_incident(self):
        with tempfile.TemporaryDirectory() as td:
            reg = id_registry.build_registry(experiments_dir=td)
            self.assertEqual(reg["families"]["EXP-FP-008x"]["lane"],
                             "ecr-replication")
            self.assertEqual(reg["families"]["EXP-FP-009x"]["lane"],
                             "transfer-dynamics")
            self.assertEqual(reg["families"]["EXP-FP-010x"]["lane"],
                             "calibration")
            incidents = [i for i in reg["incidents"]
                         if i["date"] == "2026-10-07"
                         and "EXP-FP-0050" in i["ids"]]
            self.assertEqual(len(incidents), 1)
            self.assertIn("c55fdebe", incidents[0]["lost_receipt_hash"])

    def test_build_refuses_to_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            id_registry.build_registry(experiments_dir=td)
            with self.assertRaises(id_registry.IDClaimError):
                id_registry.build_registry(experiments_dir=td)

    def test_claim_fails_closed_without_registry(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(id_registry.IDClaimError):
                id_registry.claim_id_family("EXP-FP-091x", "lane-a",
                                            experiments_dir=td)

    def test_second_claim_same_family_fails(self):
        with tempfile.TemporaryDirectory() as td:
            id_registry.build_registry(experiments_dir=td)
            rec = id_registry.claim_id_family("EXP-FP-091x", "lane-a",
                                              purpose="regression test",
                                              experiments_dir=td)
            self.assertEqual(rec["lane"], "lane-a")
            # a different lane cannot take the same family
            with self.assertRaises(id_registry.IDClaimError):
                id_registry.claim_id_family("EXP-FP-091x", "lane-b",
                                            experiments_dir=td)
            # same-lane re-claim is idempotent
            rec2 = id_registry.claim_id_family("EXP-FP-091x", "lane-a",
                                               experiments_dir=td)
            self.assertEqual(rec2["lane"], "lane-a")

    def test_concurrent_claim_simulation(self):
        """Two threads, one ID family: exactly one claim wins; the loser
        gets IDClaimError. This is the EXP-FP-0050 race, replayed."""
        import threading
        with tempfile.TemporaryDirectory() as td:
            id_registry.build_registry(experiments_dir=td)
            results = {}

            def attempt(lane):
                try:
                    id_registry.claim_id_family("EXP-FP-092x", lane,
                                                experiments_dir=td)
                    results[lane] = "ok"
                except id_registry.IDClaimError as e:
                    results[lane] = f"refused: {e}"

            ta = threading.Thread(target=attempt, args=("lane-a",))
            tb = threading.Thread(target=attempt, args=("lane-b",))
            ta.start()
            tb.start()
            ta.join()
            tb.join()
            wins = [lane for lane, r in results.items() if r == "ok"]
            refused = [lane for lane, r in results.items()
                       if r.startswith("refused")]
            self.assertEqual(len(wins), 1,
                             f"expected exactly one winner, got {results}")
            self.assertEqual(len(refused), 1,
                             f"expected exactly one refusal, got {results}")

    def test_claim_id_inside_foreign_family_fails(self):
        with tempfile.TemporaryDirectory() as td:
            id_registry.build_registry(experiments_dir=td)
            id_registry.claim_id_family("EXP-FP-093x", "lane-a",
                                        experiments_dir=td)
            with self.assertRaises(id_registry.IDClaimError):
                id_registry.claim_id("EXP-FP-0931", "lane-b",
                                     experiments_dir=td)
            # same lane can mint inside its own family
            rec = id_registry.claim_id("EXP-FP-0931", "lane-a",
                                       experiments_dir=td)
            self.assertEqual(rec["family"], "EXP-FP-093x")
            ok, info = id_registry.is_allocated("EXP-FP-0931",
                                                experiments_dir=td)
            self.assertTrue(ok)
            self.assertEqual(info["lane"], "lane-a")
