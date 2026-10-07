"""Regression test for the write/verify hash-chain convention (2026-10-07).

Bug: write_receipt chose prev_receipt_hash by mtime over ALL .json files,
including pre-chain (hashless) receipts, so prev could be None even when a
chained predecessor existed. verify_chain skips pre-chain receipts when
checking links, so it flagged a prev-mismatch. Confirmed live by the
EXP-AB-K3C worker.

Convention (now in harness): both write and verify operate on the chained
subsequence — only receipts carrying a receipt_hash, in mtime order.

These tests run on a FRESH temp receipts dir; no existing receipt is touched.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import harness  # noqa: E402


def _fake_result(exp_id):
    return {
        "experiment_id": exp_id,
        "config": {"env": "grid_world", "agent": "stateless"},
        "config_hash": "deadbeef",
        "primary_seed": 1,
        "started_utc": "2026-10-07T00:00:00+00:00",
        "episodes": [],
        "summary": {"n_episodes": 0},
    }


def _write_prechain(td, name, mtime):
    """Write a legacy pre-chain (hashless) receipt and pin its mtime."""
    p = os.path.join(td, name)
    with open(p, "w") as f:
        json.dump({"experiment_id": name, "note": "legacy, no receipt_hash"}, f)
    os.utime(p, (mtime, mtime))
    return p


def _old_style_prev(receipts_dir, exclude_path):
    """Replicates the OLD (buggy) write_receipt prev-selection logic.

    Old code: mtime-sorted over ALL .json files, take last, read its
    receipt_hash (None for a pre-chain file). Kept here to prove the new
    test would have caught the bug.
    """
    candidates = sorted(
        (q for q in os.listdir(receipts_dir) if q.endswith(".json")),
        key=lambda q: os.path.getmtime(os.path.join(receipts_dir, q)))
    candidates = [c for c in candidates
                  if os.path.join(receipts_dir, c) != exclude_path]
    if not candidates:
        return None
    with open(os.path.join(receipts_dir, candidates[-1])) as f:
        return json.load(f).get("receipt_hash")


def _write_receipt_with_prev(result, receipts_dir, prev):
    """Write a receipt body identical to write_receipt's, but with a
    caller-supplied prev — used to replay the old code path exactly."""
    receipt = {
        "experiment_id": result["experiment_id"],
        "hypothesis": "", "null_hypothesis": "", "preregistered_metric": "",
        "baseline": "", "conditions": "", "config": result["config"],
        "config_hash": result["config_hash"], "primary_seed": 1,
        "started_utc": result["started_utc"],
        "written_utc": "2026-10-07T00:00:00+00:00",
        "episodes": result["episodes"], "summary": result["summary"],
        "interpretation": "", "limitations": "",
        "prev_receipt_hash": prev,
    }
    body = json.dumps(receipt, indent=2, sort_keys=True)
    receipt["receipt_hash"] = hashlib.sha256(body.encode()).hexdigest()
    path = os.path.join(receipts_dir, f"{result['experiment_id']}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


class TestChainConvention(unittest.TestCase):
    def test_write_skips_prechain_and_agrees_with_verify(self):
        """Chained receipt A, then a NEWER pre-chain (hashless) file, then
        write_receipt for B: B must link to A's hash (not None), and
        verify_chain must accept the dir."""
        with tempfile.TemporaryDirectory() as td:
            harness.write_receipt(_fake_result("EXP-A"), td)
            a_hash = json.load(
                open(os.path.join(td, "EXP-A.json")))["receipt_hash"]
            # pre-chain file with a NEWER mtime than the chained receipt
            _write_prechain(td, "legacy.json", os.path.getmtime(
                os.path.join(td, "EXP-A.json")) + 100)
            b_path = harness.write_receipt(_fake_result("EXP-B"), td)
            b = json.load(open(b_path))
            self.assertEqual(b["prev_receipt_hash"], a_hash,
                             "new write must skip pre-chain files when "
                             "choosing prev (CHAIN CONVENTION)")
            ok, problems = harness.verify_chain(td)
            link_problems = [p for p in problems
                             if "mismatch" in p]
            self.assertEqual(link_problems, [],
                             f"write/verify disagree: {link_problems}")
            self.assertTrue(ok or all("no receipt_hash" in p
                                      for p in problems),
                            f"unexpected problems: {problems}")

    def test_old_code_path_fails_verify(self):
        """The pre-fix write logic produces a receipt that verify_chain
        flags — this proves the regression test catches the original bug."""
        with tempfile.TemporaryDirectory() as td:
            harness.write_receipt(_fake_result("EXP-A"), td)
            _write_prechain(td, "legacy.json", os.path.getmtime(
                os.path.join(td, "EXP-A.json")) + 100)
            old_prev = _old_style_prev(
                td, os.path.join(td, "EXP-B.json"))
            self.assertIsNone(old_prev,
                              "old logic yields prev=None here (chained to "
                              "the hashless file)")
            _write_receipt_with_prev(_fake_result("EXP-B"), td, old_prev)
            ok, problems = harness.verify_chain(td)
            self.assertFalse(ok)
            self.assertTrue(any("prev_receipt_hash mismatch" in p
                                for p in problems),
                            f"old path should trip prev-mismatch: {problems}")

    def test_first_chained_receipt_after_only_prechain_has_prev_none(self):
        """With only pre-chain files present, the first chained receipt
        gets prev=None and verify accepts the linkage."""
        with tempfile.TemporaryDirectory() as td:
            _write_prechain(td, "legacy1.json", 1000.0)
            _write_prechain(td, "legacy2.json", 2000.0)
            b_path = harness.write_receipt(_fake_result("EXP-B"), td)
            b = json.load(open(b_path))
            self.assertIsNone(b["prev_receipt_hash"])
            ok, problems = harness.verify_chain(td)
            self.assertEqual([p for p in problems if "mismatch" in p], [])

    def test_three_chained_receipts_link_in_order(self):
        """Sanity: consecutive writes chain A -> B -> C and verify passes."""
        with tempfile.TemporaryDirectory() as td:
            harness.write_receipt(_fake_result("EXP-A"), td)
            harness.write_receipt(_fake_result("EXP-B"), td)
            harness.write_receipt(_fake_result("EXP-C"), td)
            recs = {n: json.load(open(os.path.join(td, f"EXP-{n}.json")))
                    for n in "ABC"}
            self.assertIsNone(recs["A"]["prev_receipt_hash"])
            self.assertEqual(recs["B"]["prev_receipt_hash"],
                             recs["A"]["receipt_hash"])
            self.assertEqual(recs["C"]["prev_receipt_hash"],
                             recs["B"]["receipt_hash"])
            ok, problems = harness.verify_chain(td)
            self.assertTrue(ok, problems)


if __name__ == "__main__":
    unittest.main()
