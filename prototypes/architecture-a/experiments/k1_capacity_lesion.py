"""K1 — CAPACITY LESION (Architecture A load-bearing claim 1).

Hypothesis: a bounded workspace (K=3) creates measurable competition:
  two simultaneous high-salience items interfere (the loser is evicted
  and never reaches broadcast); with unbounded capacity (K=inf) the
  interference vanishes.
Null: target survival is identical at K=3 and K=inf (the bound
  contributes nothing).
Preregistered metric: interference index I = P(target lost | K=3)
  - P(target lost | K=inf), per distractor-bid condition. PASS requires
  I > 0.40 in the matched-bid condition AND monotonic increase of
  P(target lost | K=3) with distractor bid (competitive, not random).
Baseline: K=inf (unbounded buffer, same eviction policy object).
Ablation: the K=3 condition IS the ablation of capacity.
Procedure: pre-fill buffer with 2 high-salience residents (bid 0.90,
  an ongoing task set); then admit target (bid 0.85) and distractor
  (bid in {0.50, 0.85, 0.95}) in random order; record whether the
  target survives (admitted and never evicted). 400 trials/condition,
  seeded. Consumers declared = full registry (constant).
Seed: 20261007.
Kill: no K3/Kinf difference -> DELETE the buffer, keep a plain queue.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workspace_buffer import BoundedWorkspace, admit_swallowing_refusal
from broadcast import CONSUMER_REGISTRY

SEED = 20261007
TRIALS = 400
TARGET_BID = 0.85
RESIDENT_BID = 0.90
DISTRACTOR_BIDS = (0.50, 0.85, 0.95)


def run_condition(capacity, distractor_bid, seed):
    rng = random.Random(seed)
    lost = 0
    for _ in range(TRIALS):
        buf = BoundedWorkspace(capacity,
                               consumer_registry=set(CONSUMER_REGISTRY))
        # ongoing task set: two high-salience residents
        for i in range(2):
            admit_swallowing_refusal(
                buf, kind="resident", payload={"slot": i}, bid=RESIDENT_BID,
                declared_consumers=list(CONSUMER_REGISTRY))
        # target + distractor arrive together, random order
        arrivals = [("target", TARGET_BID), ("distractor", distractor_bid)]
        rng.shuffle(arrivals)
        target_id = None
        for kind, bid in arrivals:
            rec = admit_swallowing_refusal(
                buf, kind=kind, payload={}, bid=bid,
                declared_consumers=list(CONSUMER_REGISTRY))
            if kind == "target":
                target_id = rec["item_id"]
        live = {it.item_id for it in buf.contents()}
        if target_id not in live:
            lost += 1
    return lost / TRIALS


def main():
    print("K1 — capacity lesion: dual-task interference at K=3 vs K=inf")
    print(f"hypothesis: bounded K creates competition; null: no difference")
    print(f"trials/condition={TRIALS} seed={SEED}")
    results = {}
    for d_bid in DISTRACTOR_BIDS:
        p3 = run_condition(3, d_bid, SEED + int(d_bid * 100))
        pinf = run_condition(None, d_bid, SEED + int(d_bid * 100))
        interference = p3 - pinf
        results[str(d_bid)] = {"p_lost_K3": p3, "p_lost_Kinf": pinf,
                               "interference": interference}
        print(f"  distractor bid={d_bid}: P(lost|K=3)={p3:.3f} "
              f"P(lost|K=inf)={pinf:.3f} I={interference:+.3f}")
    mono = (results["0.5"]["p_lost_K3"] <= results["0.85"]["p_lost_K3"]
            <= results["0.95"]["p_lost_K3"])
    gate = results["0.85"]["interference"] > 0.40 and mono
    verdict = ("PASS — interference present at K=3, absent at K=inf; "
               "monotonic in distractor bid (competitive)"
               if gate else
               "FAIL — kill condition met: no capacity-dependent "
               "interference; DELETE the buffer")
    print("monotonic(K=3):", mono)
    print("VERDICT:", verdict)
    receipt = {"experiment": "K1_capacity_lesion", "seed": SEED,
               "trials": TRIALS, "target_bid": TARGET_BID,
               "resident_bid": RESIDENT_BID, "results": results,
               "monotonic": mono, "pass": bool(gate), "verdict": verdict}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "receipts", "k1_capacity_lesion.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
