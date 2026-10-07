"""REPRODUCTION driver — Architecture A K1-K4 on 5 fresh seeds each.

Runs the ORIGINAL experiment code paths (imports the original experiment
modules; no copies) with fresh seeds distinct from the originals
(originals: K1/K2/K3 seed 20261007; K4 seeds 11/22/33/44).
Verdict per seed uses the ORIGINAL preregistered gates. A kill/experiment
REPRODUCES if the verdict holds on >=4/5 seeds.
Receipts: receipts/repro_kN_*.json (original schema + reproduction fields).
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import k1_capacity_lesion as K1
import k2_broadcast_lesion as K2
import k3_ignition_probe as K3
import k4_attention_baseline as K4
from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists
from broadcast import CONSUMER_REGISTRY

RECEIPTS = os.path.join(HERE, "..", "receipts")

SEEDS = {
    "K1": [71001, 71002, 71003, 71004, 71005],
    "K2": [71101, 71102, 71103, 71104, 71105],
    "K3": [71201, 71202, 71203, 71204, 71205],
    "K4": [71301, 71302, 71303, 71304, 71305],
}


def write_receipt(name, receipt):
    os.makedirs(RECEIPTS, exist_ok=True)
    path = os.path.join(RECEIPTS, f"repro_{name}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------- K1
def repro_k1():
    per_seed = {}
    for seed in SEEDS["K1"]:
        results = {}
        for d_bid in K1.DISTRACTOR_BIDS:
            p3 = K1.run_condition(3, d_bid, seed + int(d_bid * 100))
            pinf = K1.run_condition(None, d_bid, seed + int(d_bid * 100))
            results[str(d_bid)] = {"p_lost_K3": p3, "p_lost_Kinf": pinf,
                                   "interference": p3 - pinf}
        mono = (results["0.5"]["p_lost_K3"] <= results["0.85"]["p_lost_K3"]
                <= results["0.95"]["p_lost_K3"])
        gate = results["0.85"]["interference"] > 0.40 and mono
        per_seed[str(seed)] = {"results": results, "monotonic": mono,
                               "pass": bool(gate)}
        print(f"  K1 seed={seed}: I@0.85={results['0.85']['interference']:.3f} "
              f"mono={mono} -> {'PASS' if gate else 'FAIL'}")
    passes = sum(1 for v in per_seed.values() if v["pass"])
    repro = passes >= 4
    receipt = {"experiment": "K1_capacity_lesion", "reproduction": True,
               "original_seed": K1.SEED, "seeds": SEEDS["K1"],
               "trials": K1.TRIALS, "per_seed": per_seed,
               "passes": passes, "reproduces": repro,
               "verdict": f"REPRODUCES {passes}/5 seeds" if repro else
                          f"FAILS TO REPRODUCE — only {passes}/5 seeds pass"}
    path = write_receipt("k1_capacity_lesion", receipt)
    print(f"K1: {passes}/5 -> {'REPRODUCES' if repro else 'NO-REPRO'}\n  {path}")
    return repro


# ---------------------------------------------------------------- K2
def k2_one_seed(seed):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       ignition_kwargs={"theta": 0.45})
    env = ChangingRelevanceEnv(channels, 400, seed=seed)
    d_base, _, _ = K2.run_phase(wk, env, K2.TICKS)
    if not all(v > 0 for v in d_base.values()):
        return {"pass": False, "reason": "baseline did not deliver to every consumer",
                "baseline": d_base}
    wk.bus.lesion_consumer("memory_admit")
    d_sel, _, _ = K2.run_phase(wk, env, K2.TICKS)
    wk.bus.restore_consumer("memory_admit")

    def approx(a, b, tol=0.15):
        return abs(a - b) <= tol * max(b, 1)
    sel_ok = (d_sel["memory_admit"] == 0 and
              all(approx(d_sel[k], d_base[k]) for k in d_sel
                  if k != "memory_admit"))
    wk.bus.lesion_all()
    d_tot, adm_tot, ign_tot = K2.run_phase(wk, env, K2.TICKS)
    wk.bus.restore_all()
    tot_ok = (all(v == 0 for v in d_tot.values())
              and adm_tot > 0 and ign_tot > 0)
    d_rec, _, _ = K2.run_phase(wk, env, K2.TICKS)
    rec_ok = all(abs(d_rec[k] - d_base[k]) <= 0.15 * max(d_base[k], 1)
                 for k in d_rec)
    return {"baseline": d_base, "selective_lesion_deltas": d_sel,
            "total_lesion_deltas": d_tot,
            "internal_admissions_during_total_lesion": adm_tot,
            "internal_ignitions_during_total_lesion": ign_tot,
            "recovery_deltas": d_rec, "selective_ok": sel_ok,
            "total_ok": tot_ok, "recovery_ok": rec_ok,
            "pass": bool(sel_ok and tot_ok and rec_ok)}


def repro_k2():
    per_seed = {}
    for seed in SEEDS["K2"]:
        try:
            r = k2_one_seed(seed)
        except Exception as ex:
            r = {"pass": False, "reason": f"{type(ex).__name__}: {ex}",
                 "trace": traceback.format_exc(limit=3)}
        per_seed[str(seed)] = r
        print(f"  K2 seed={seed}: sel={r.get('selective_ok')} "
              f"tot={r.get('total_ok')} rec={r.get('recovery_ok')} -> "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    passes = sum(1 for v in per_seed.values() if v["pass"])
    repro = passes >= 4
    receipt = {"experiment": "K2_broadcast_lesion", "reproduction": True,
               "original_seed": K2.SEED, "seeds": SEEDS["K2"],
               "ticks_per_phase": K2.TICKS, "per_seed": per_seed,
               "passes": passes, "reproduces": repro,
               "verdict": f"REPRODUCES {passes}/5 seeds" if repro else
                          f"FAILS TO REPRODUCE — only {passes}/5 seeds pass"}
    path = write_receipt("k2_broadcast_lesion", receipt)
    print(f"K2: {passes}/5 -> {'REPRODUCES' if repro else 'NO-REPRO'}\n  {path}")
    return repro


# ---------------------------------------------------------------- K3
def repro_k3():
    per_seed = {}
    for seed in SEEDS["K3"]:
        real = K3.sweep(False, seed)
        graded = K3.sweep(True, seed + 1)
        w = K3.transition_width(real)
        p_low = (sum(p for x, p in real if x <= K3.THETA - 0.25) /
                 max(1, sum(1 for x, p in real if x <= K3.THETA - 0.25)))
        p_high = (sum(p for x, p in real if x >= K3.THETA + 0.25) /
                  max(1, sum(1 for x, p in real if x >= K3.THETA + 0.25)))
        lin_err = sum(abs(p - x) for x, p in graded) / len(graded)
        gate = (w < 0.12 and p_low == 0.0 and p_high == 1.0 and lin_err < 0.08)
        per_seed[str(seed)] = {"transition_width": w, "p_below": p_low,
                               "p_above": p_high,
                               "graded_control_linearity_error": lin_err,
                               "pass": bool(gate)}
        print(f"  K3 seed={seed}: W={w:.3f} p_low={p_low:.3f} "
              f"p_high={p_high:.3f} lin_err={lin_err:.4f} -> "
              f"{'PASS' if gate else 'FAIL'}")
    passes = sum(1 for v in per_seed.values() if v["pass"])
    repro = passes >= 4
    receipt = {"experiment": "K3_ignition_linearity_probe", "reproduction": True,
               "original_seed": K3.SEED, "seeds": SEEDS["K3"],
               "theta": K3.THETA, "per_seed": per_seed,
               "passes": passes, "reproduces": repro,
               "verdict": f"REPRODUCES {passes}/5 seeds" if repro else
                          f"FAILS TO REPRODUCE — only {passes}/5 seeds pass"}
    path = write_receipt("k3_ignition_probe", receipt)
    print(f"K3: {passes}/5 -> {'REPRODUCES' if repro else 'NO-REPRO'}\n  {path}")
    return repro


# ---------------------------------------------------------------- K4
def repro_k4():
    per_seed = {}
    for seed in SEEDS["K4"]:
        tl, p2l, al, gl = K4.run(False, seed)
        tf, p2f, af, gf = K4.run(True, seed)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= K4.MARGIN
        per_seed[str(seed)] = {"learned_total": round(tl, 2),
                               "fixed_total": round(tf, 2),
                               "ratio": round(r, 3), "win": bool(win),
                               "learned_p2": round(p2l, 2),
                               "fixed_p2": round(p2f, 2),
                               "learned_gains": {k: round(v, 3)
                                                 for k, v in gl.items()}}
        print(f"  K4 seed={seed}: learned={tl:.1f} fixed={tf:.1f} "
              f"R={r:.2f} -> {'WIN' if win else 'LOSS'}")
    wins = sum(1 for v in per_seed.values() if v["win"])
    repro = wins >= 4
    receipt = {"experiment": "K4_attention_baseline", "reproduction": True,
               "original_seeds": list(K4.SEEDS), "seeds": SEEDS["K4"],
               "ticks": K4.TICKS, "theta": K4.THETA, "margin": K4.MARGIN,
               "per_seed": per_seed, "wins": wins, "reproduces": repro,
               "verdict": f"REPRODUCES {wins}/5 seeds" if repro else
                          f"FAILS TO REPRODUCE — only {wins}/5 seeds win"}
    path = write_receipt("k4_attention_baseline", receipt)
    print(f"K4: {wins}/5 -> {'REPRODUCES' if repro else 'NO-REPRO'}\n  {path}")
    return repro


if __name__ == "__main__":
    print("=== Architecture A multi-seed reproduction ===")
    out = {}
    out["K1"] = repro_k1()
    out["K2"] = repro_k2()
    out["K3"] = repro_k3()
    out["K4"] = repro_k4()
    print("\n=== REPRODUCTION SUMMARY ===")
    for k, v in out.items():
        print(f"A-{k}: {'REPRODUCES' if v else 'FAILS TO REPRODUCE'}")
