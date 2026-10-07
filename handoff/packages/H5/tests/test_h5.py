"""H5 package test — consolidation_cycle (S-01, the EXP-FP-0005 winner).

Standalone proof from the packaged source (mechanism-level):
  1. Cycle: decay -> greedy merge -> prune over a synthetic episode
     snapshot; produces a summary layer + promotion list.
  2. Determinism: two runs over the same input are identical.
  3. Input immutability: the caller's episode list is never mutated and
     no source episode is deleted (prune touches the summary layer only).
  4. Merge correctness: near-duplicate vectors merge into one summary;
     dissimilar vectors stay separate.
  5. Verification flow: summaries start UNVERIFIED; mark_verified flips
     them to VERIFIED.
  6. Fail-closed: malformed entries / bad config raise CycleRefused.

Note the honest bound (EXP-FP-0005): mechanism behaves per docs, but
offline replay showed no measured performance lift vs no-replay on the
pomaze corpus, and PE-magnitude prioritization lost to uniform replay.
This test proves the MECHANISM, not a performance lift.

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from consolidation_cycle_s01 import (
    ConsolidationCycle, CycleConfig, CycleRefused, adapt_episode,
    UNVERIFIED, VERIFIED)


def make_episodes():
    # two clusters: near-duplicate vectors (merge) + one outlier (separate)
    eps = []
    for i in range(6):
        eps.append({"episode_id": f"ep-a-{i}",
                    "context_vector": [1.0 + 0.01 * i, 0.0 + 0.01 * i],
                    "salience": 0.9})
    for i in range(6):
        eps.append({"episode_id": f"ep-b-{i}",
                    "context_vector": [0.0 + 0.01 * i, 1.0 - 0.01 * i],
                    "salience": 0.8})
    eps.append({"episode_id": "ep-out",
                "context_vector": [10.0, -10.0],
                "salience": 0.95})
    return [adapt_episode(e) for e in eps]


def main():
    results, ok_all = {}, True

    # 1+2. cycle + determinism
    cyc = ConsolidationCycle(CycleConfig(promotion_k=4))
    entries = make_episodes()
    snapshot = copy.deepcopy(entries)
    r1 = cyc.run(entries, tick=1)
    r2 = cyc.run(make_episodes(), tick=1)
    det = (r1.summaries == r2.summaries and r1.promotion == r2.promotion)
    n_sum = len(r1.summaries)
    ok = r1.ran and n_sum == 3 and det and len(r1.promotion) > 0
    results["cycle_determinism"] = {"ok": ok, "n_summaries": n_sum,
                                   "promotion": r1.promotion,
                                   "deterministic": det}
    print(f"  cycle: ran={r1.ran} summaries={n_sum} (need 3: 2 clusters + "
          f"outlier) promotion={len(r1.promotion)} deterministic={det} -> "
          f"{'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 3. input immutability
    ok = entries == snapshot
    results["input_immutable"] = {"ok": ok}
    print(f"  input immutability: {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 4. merge correctness: cluster members share one summary
    member_sets = [set(s["member_ids"]) for s in r1.summaries]
    a_ids = {f"ep-a-{i}" for i in range(6)}
    b_ids = {f"ep-b-{i}" for i in range(6)}
    merged_a = any(a_ids <= ms for ms in member_sets)
    merged_b = any(b_ids <= ms for ms in member_sets)
    out_alone = any(ms == {"ep-out"} for ms in member_sets)
    ok = merged_a and merged_b and out_alone
    results["merge_correctness"] = {"ok": ok, "cluster_a_merged": merged_a,
                                   "cluster_b_merged": merged_b,
                                   "outlier_alone": out_alone}
    print(f"  merge: cluster_a={merged_a} cluster_b={merged_b} "
          f"outlier_alone={out_alone} -> {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 5. verification flow
    statuses = {s["status"] for s in r1.summaries}
    sid = r1.summaries[0]["summary_id"]
    r1.mark_verified(sid, evidence="package test")
    after = [s["status"] for s in r1.summaries
             if s["summary_id"] == sid][0]
    ok = statuses == {UNVERIFIED} and after == VERIFIED
    results["verification_flow"] = {"ok": ok, "initial": sorted(statuses),
                                    "after_mark": after}
    print(f"  verification: initial={sorted(statuses)} after_mark={after} "
          f"-> {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 6. fail-closed
    refused = 0
    try:
        adapt_episode({"episode_id": "", "context_vector": [1.0]})
    except CycleRefused:
        refused += 1
    try:
        CycleConfig(promotion_k=-1)
    except CycleRefused:
        refused += 1
    try:
        cyc.run([{"id": "x"}], tick=1)  # missing vector
    except CycleRefused:
        refused += 1
    ok = refused == 3
    results["fail_closed"] = {"ok": ok, "refused": refused}
    print(f"  fail-closed: refused {refused}/3 -> {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    print("OVERALL ->", "PASS" if ok_all else "FAIL")
    receipt = {"package": "H5", "test": "test_h5.py", "checks": results,
               "pass": bool(ok_all),
               "bound": ("mechanism proven; offline performance lift "
                         "UNPROVEN per EXP-FP-0005 (bound, not a lift claim)"),
               "verdict": "PASS" if ok_all else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
