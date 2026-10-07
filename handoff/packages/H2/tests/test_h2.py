"""H2 package test — bounded workspace buffer + sole-path broadcast.

Standalone proof from the packaged sources:
  K1 (capacity lesion): interference index I = P(target lost|K=3) -
    P(target lost|K=inf) > 0.40 in the matched-bid condition AND monotone
    increase of P(lost|K=3) with distractor bid; exactly 0 at K=inf.
  K2 (broadcast lesion): selective lesion silences only the lesioned
    consumer; total lesion silences ALL six consumers at once while internal
    processing (admissions, ignition cycles) continues; full recovery;
    no side channels.

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only. Seeded.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from workspace_buffer import BoundedWorkspace, admit_swallowing_refusal
from broadcast import CONSUMER_REGISTRY
from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists

K1_SEED = 20261007
K1_TRIALS = 400
K2_SEED = 20261007
K2_TICKS = 30


# ---------------- K1 ----------------
def k1_condition(capacity, distractor_bid, seed):
    rng = random.Random(seed)
    lost = 0
    for _ in range(K1_TRIALS):
        buf = BoundedWorkspace(capacity,
                               consumer_registry=set(CONSUMER_REGISTRY))
        for i in range(2):
            admit_swallowing_refusal(
                buf, kind="resident", payload={"slot": i}, bid=0.90,
                declared_consumers=list(CONSUMER_REGISTRY))
        arrivals = [("target", 0.85), ("distractor", distractor_bid)]
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
    return lost / K1_TRIALS


def run_k1():
    results = {}
    for d_bid in (0.50, 0.85, 0.95):
        p3 = k1_condition(3, d_bid, K1_SEED + int(d_bid * 100))
        pinf = k1_condition(None, d_bid, K1_SEED + int(d_bid * 100))
        results[str(d_bid)] = {"p_lost_K3": p3, "p_lost_Kinf": pinf,
                               "interference": p3 - pinf}
        print(f"  distractor={d_bid}: P(lost|K=3)={p3:.3f} "
              f"P(lost|K=inf)={pinf:.3f} I={p3 - pinf:+.3f}")
    mono = (results["0.5"]["p_lost_K3"] <= results["0.85"]["p_lost_K3"]
            <= results["0.95"]["p_lost_K3"])
    gate = results["0.85"]["interference"] > 0.40 and mono
    print("  K1 monotonic:", mono, "->", "PASS" if gate else "FAIL")
    return gate, results


# ---------------- K2 ----------------
def consumer_effects(wk):
    return {name: sum(1 for d in wk.bus.deliveries
                     if d["consumer"] == name and d["status"] == "delivered")
            for name in CONSUMER_REGISTRY}


def run_phase(wk, env, ticks):
    before = consumer_effects(wk)
    adm_before = wk.buffer.admitted_total
    ign_before = len(wk.ignition.history)
    for _ in range(ticks):
        wk.step(env.observe(), env)
    after = consumer_effects(wk)
    return ({k: after[k] - before[k] for k in after},
            wk.buffer.admitted_total - adm_before,
            len(wk.ignition.history) - ign_before)


def run_k2():
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       ignition_kwargs={"theta": 0.45})
    env = ChangingRelevanceEnv(channels, 400, seed=K2_SEED)
    d_base, _, _ = run_phase(wk, env, K2_TICKS)
    print("  baseline deliveries:", d_base)
    assert all(v > 0 for v in d_base.values()), \
        "baseline must deliver to every consumer"

    wk.bus.lesion_consumer("memory_admit")
    d_sel, _, _ = run_phase(wk, env, K2_TICKS)
    wk.bus.restore_consumer("memory_admit")
    sel_ok = (d_sel["memory_admit"] == 0 and all(
        abs(d_sel[k] - d_base[k]) <= 0.15 * max(d_base[k], 1)
        for k in d_sel if k != "memory_admit"))
    print("  selective lesion ok:", sel_ok)

    wk.bus.lesion_all()
    d_tot, adm_tot, ign_tot = run_phase(wk, env, K2_TICKS)
    wk.bus.restore_all()
    tot_ok = (all(v == 0 for v in d_tot.values())
              and adm_tot > 0 and ign_tot > 0)
    print(f"  total lesion: all-zero={all(v == 0 for v in d_tot.values())} "
          f"internal admissions +{adm_tot} ignitions +{ign_tot} -> {tot_ok}")

    d_rec, _, _ = run_phase(wk, env, K2_TICKS)
    rec_ok = all(abs(d_rec[k] - d_base[k]) <= 0.15 * max(d_base[k], 1)
                 for k in d_rec)
    print("  recovery ok:", rec_ok)
    gate = sel_ok and tot_ok and rec_ok
    print("  K2 ->", "PASS" if gate else "FAIL")
    return gate, {"baseline": d_base, "selective_ok": sel_ok,
                  "total_ok": tot_ok, "recovery_ok": rec_ok}


def main():
    print("K1 — capacity lesion")
    g1, r1 = run_k1()
    print("K2 — broadcast lesion (sole-path)")
    g2, r2 = run_k2()
    gate = g1 and g2
    print("OVERALL ->", "PASS" if gate else "FAIL")
    receipt = {"package": "H2", "test": "test_h2.py",
               "K1": {"pass": bool(g1), "results": r1},
               "K2": {"pass": bool(g2), "results": r2},
               "pass": bool(gate),
               "verdict": "PASS" if gate else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
