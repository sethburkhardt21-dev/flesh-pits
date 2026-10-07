"""SECOND-LANE independent replication of Architecture A K1-K4.

Preregistered: flesh-pits/prototypes/architecture-a/receipts/prereg_repro_second_lane.json
(preregistered_utc 2026-10-07T07:49:00Z, sealed BEFORE this run).

Independence: this driver is written from the K1-K4 spec; it imports ONLY the
architecture modules under test (workspace_buffer, broadcast, ignition, tick,
envs). It does NOT import k1_capacity_lesion, k2_broadcast_lesion,
k3_ignition_probe, k4_attention_baseline, or repro_multiseed_a.

R1 wiring: the run exercises the CURRENT default WorkspaceTick (R1
sub-ignition exploratory path is the default _select_action since K8). K1/K3
do not use the tick at all. K2/K4 additionally record how often the
sub-ignition path fired (behavioral check, not gated).

Reproduction rule: REPRODUCES iff the original per-seed gate holds on >=4/5
fresh seeds. Failure on >=2/5 seeds is a NON-REPRODUCTION.
"""
import json
import os
import random
import sys
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from workspace_buffer import BoundedWorkspace, admit_swallowing_refusal
from broadcast import CONSUMER_REGISTRY
from ignition import RecurrentIgnition
from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists

RECEIPTS = os.path.join(PARENT, "receipts")
SEED_SETS = {
    "K1": [72001, 72002, 72003, 72004, 72005],
    "K2": [72101, 72102, 72103, 72104, 72105],
    "K3": [72201, 72202, 72203, 72204, 72205],
    "K4": [72301, 72302, 72303, 72304, 72305],
}
PREREG_PATH = os.path.join(RECEIPTS, "prereg_repro_second_lane.json")


def seal(body):
    """Hash-chained seal: receipt_hash over the body (sort_keys, indent 2)."""
    core = {k: v for k, v in body.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()


def store(name, receipt):
    os.makedirs(RECEIPTS, exist_ok=True)
    path = os.path.join(RECEIPTS, name)
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


# ================================================================ K1
BID_LEVELS = (0.50, 0.85, 0.95)
K1_TRIALS = 400
TARGET_STRENGTH = 0.85
RESIDENT_STRENGTH = 0.90


def k1_loss_rate(capacity, distractor_bid, subseed):
    """P(target lost): pre-fill 2 high-salience residents, admit target +
    distractor in random order, check whether the target item survives."""
    rng = random.Random(subseed)
    registry = list(CONSUMER_REGISTRY)
    lost = 0
    for _ in range(K1_TRIALS):
        buf = BoundedWorkspace(capacity, consumer_registry=set(registry))
        for i in range(2):
            admit_swallowing_refusal(
                buf, kind="resident", payload={"slot": i},
                bid=RESIDENT_STRENGTH, declared_consumers=registry)
        arrivals = [("target", TARGET_STRENGTH),
                    ("distractor", distractor_bid)]
        rng.shuffle(arrivals)
        target_id = None
        for kind, bid in arrivals:
            rec = admit_swallowing_refusal(
                buf, kind=kind, payload={}, bid=bid,
                declared_consumers=registry)
            if kind == "target":
                target_id = rec["item_id"]
        if target_id not in {it.item_id for it in buf.contents()}:
            lost += 1
    return lost / K1_TRIALS


def lane_k1():
    per_seed = {}
    for seed in SEED_SETS["K1"]:
        rows = {}
        for d in BID_LEVELS:
            sub = seed + int(d * 100)
            bounded = k1_loss_rate(3, d, sub)
            unbounded = k1_loss_rate(None, d, sub)
            rows[str(d)] = {"p_lost_K3": bounded, "p_lost_Kinf": unbounded,
                            "interference": bounded - unbounded}
        keys = [str(d) for d in BID_LEVELS]
        mono = all(rows[keys[i]]["p_lost_K3"] <= rows[keys[i + 1]]["p_lost_K3"]
                   for i in range(len(keys) - 1))
        gate = rows["0.85"]["interference"] > 0.40 and mono
        per_seed[str(seed)] = {"conditions": rows, "monotonic": mono,
                               "pass": bool(gate)}
        print(f"  K1 seed={seed}: I@0.85={rows['0.85']['interference']:.3f} "
              f"mono={mono} -> {'PASS' if gate else 'FAIL'}")
    return per_seed


# ================================================================ K2
K2_TICKS = 30
SUB_IGN_PATH = "sub_ignition_explore"


def k2_effects(wk):
    return {name: sum(1 for d in wk.bus.deliveries
                      if d["consumer"] == name and d["status"] == "delivered")
            for name in CONSUMER_REGISTRY}


def k2_phase(wk, env, ticks):
    before = k2_effects(wk)
    adm0 = wk.buffer.admitted_total
    ign0 = len(wk.ignition.history)
    sub = 0
    for _ in range(ticks):
        trace = wk.step(env.observe(), env)
        if trace.get("action_selection", {}).get("path") == SUB_IGN_PATH:
            sub += 1
    after = k2_effects(wk)
    return ({k: after[k] - before[k] for k in after},
            wk.buffer.admitted_total - adm0,
            len(wk.ignition.history) - ign0, sub)


def k2_within(a, b, tol=0.15):
    return abs(a - b) <= tol * max(b, 1)


def lane_k2_seed(seed):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       ignition_kwargs={"theta": 0.45})
    env = ChangingRelevanceEnv(channels, 400, seed=seed)
    base, _, _, sub_base = k2_phase(wk, env, K2_TICKS)
    if not all(v > 0 for v in base.values()):
        return {"pass": False,
                "reason": "baseline did not deliver to every consumer",
                "baseline": base}

    wk.bus.lesion_consumer("memory_admit")
    sel, _, _, sub_sel = k2_phase(wk, env, K2_TICKS)
    wk.bus.restore_consumer("memory_admit")
    selective_ok = (sel["memory_admit"] == 0 and
                    all(k2_within(sel[k], base[k]) for k in sel
                        if k != "memory_admit"))

    wk.bus.lesion_all()
    tot, adm_tot, ign_tot, sub_tot = k2_phase(wk, env, K2_TICKS)
    wk.bus.restore_all()
    total_ok = (all(v == 0 for v in tot.values())
                and adm_tot > 0 and ign_tot > 0)

    rec, _, _, sub_rec = k2_phase(wk, env, K2_TICKS)
    recovery_ok = all(k2_within(rec[k], base[k]) for k in rec)

    return {"baseline": base, "selective_deltas": sel,
            "total_deltas": tot, "recovery_deltas": rec,
            "admissions_during_total_lesion": adm_tot,
            "ignitions_during_total_lesion": ign_tot,
            "selective_ok": selective_ok, "total_ok": total_ok,
            "recovery_ok": recovery_ok,
            "sub_ignition_actions_per_phase": {
                "baseline": sub_base, "selective_lesion": sub_sel,
                "total_lesion": sub_tot, "recovery": sub_rec},
            "pass": bool(selective_ok and total_ok and recovery_ok)}


def lane_k2():
    per_seed = {}
    for seed in SEED_SETS["K2"]:
        r = lane_k2_seed(seed)
        per_seed[str(seed)] = r
        print(f"  K2 seed={seed}: sel={r.get('selective_ok')} "
              f"tot={r.get('total_ok')} rec={r.get('recovery_ok')} -> "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    return per_seed


# ================================================================ K3
K3_POINTS = 51
K3_TRIALS = 200
K3_JITTER = 0.03
K3_THETA = 0.6


def k3_sweep(linear_probe, seed):
    rng = random.Random(seed)
    curve = []
    for i in range(K3_POINTS):
        x = i / (K3_POINTS - 1)
        hits = 0.0
        for _ in range(K3_TRIALS):
            ig = RecurrentIgnition(theta=K3_THETA, linear_probe=linear_probe)
            rec = ig.ignite(x, jitter=rng.gauss(0, K3_JITTER))
            hits += rec["strength"] if linear_probe else (
                1.0 if rec["ignited"] else 0.0)
        curve.append((x, hits / K3_TRIALS))
    return curve


def k3_width(curve):
    try:
        lo = next(x for x, p in curve if p >= 0.05)
        hi = next(x for x, p in curve if p >= 0.95)
    except StopIteration:
        return float("inf")
    return hi - lo


def lane_k3():
    per_seed = {}
    for seed in SEED_SETS["K3"]:
        real = k3_sweep(False, seed)
        graded = k3_sweep(True, seed + 1)
        w = k3_width(real)
        lo_pts = [p for x, p in real if x <= K3_THETA - 0.25]
        hi_pts = [p for x, p in real if x >= K3_THETA + 0.25]
        p_lo = sum(lo_pts) / len(lo_pts)
        p_hi = sum(hi_pts) / len(hi_pts)
        lin_err = sum(abs(p - x) for x, p in graded) / len(graded)
        gate = (w < 0.12 and p_lo == 0.0 and p_hi == 1.0 and lin_err < 0.08)
        per_seed[str(seed)] = {"transition_width": w, "p_below": p_lo,
                               "p_above": p_hi,
                               "graded_control_linearity_error": lin_err,
                               "pass": bool(gate)}
        print(f"  K3 seed={seed}: W={w:.3f} p_lo={p_lo:.3f} "
              f"p_hi={p_hi:.3f} lin_err={lin_err:.4f} -> "
              f"{'PASS' if gate else 'FAIL'}")
    return per_seed


# ================================================================ K4
K4_TICKS = 200
K4_MARGIN = 1.30


def k4_run(frozen, seed):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": 0.45})
    env = ChangingRelevanceEnv(channels, K4_TICKS, seed=seed,
                               stationary_signals=True)
    sub = 0
    for _ in range(K4_TICKS):
        trace = wk.step(env.observe(), env)
        if trace.get("action_selection", {}).get("path") == SUB_IGN_PATH:
            sub += 1
    return sum(wk.rewards), sub, dict(wk.arbitrator.gains)


def lane_k4():
    per_seed = {}
    for seed in SEED_SETS["K4"]:
        tl, subl, gl = k4_run(False, seed)
        tf, subf, _ = k4_run(True, seed)
        ratio = tl / tf if tf > 0 else float("inf")
        win = ratio >= K4_MARGIN
        per_seed[str(seed)] = {"learned_total": round(tl, 2),
                               "fixed_total": round(tf, 2),
                               "ratio": round(ratio, 3), "win": bool(win),
                               "sub_ignition_actions": {
                                   "learned": subl, "frozen": subf},
                               "learned_gains": {k: round(v, 3)
                                                 for k, v in gl.items()}}
        print(f"  K4 seed={seed}: learned={tl:.1f} fixed={tf:.1f} "
              f"R={ratio:.2f} -> {'WIN' if win else 'LOSS'}")
    return per_seed


# ================================================================ main
def verdict_line(results, key):
    n = sum(1 for v in results.values() if v[key])
    ok = n >= 4
    return n, ok, (f"REPRODUCES {n}/5 seeds" if ok
                   else f"FAILS TO REPRODUCE — only {n}/5 seeds hold")


def main():
    prereg = json.load(open(PREREG_PATH))
    chain = [prereg["receipt_hash"]]
    summary = {}

    print("== second lane: K1 capacity lesion ==")
    k1 = lane_k1()
    n, ok, line = verdict_line(k1, "pass")
    rec = {"experiment": "K1_capacity_lesion", "lane": "second",
           "reproduction": True, "original_seed": 20261007,
           "seeds": SEED_SETS["K1"], "trials_per_condition": K1_TRIALS,
           "per_seed": k1, "passes": n, "reproduces": ok, "verdict": line,
           "prereg": {"path": PREREG_PATH,
                      "receipt_hash": prereg["receipt_hash"],
                      "preregistered_utc": prereg["preregistered_utc"]},
           "prev_receipt_hash": chain[-1]}
    rec["receipt_hash"] = seal(rec)
    path = store("repro_second_lane_k1_capacity_lesion.json", rec)
    chain.append(rec["receipt_hash"])
    print(f"K1: {line}\n  {path}")
    summary["K1"] = ok

    print("== second lane: K2 broadcast lesion ==")
    k2 = lane_k2()
    n, ok, line = verdict_line(k2, "pass")
    rec = {"experiment": "K2_broadcast_lesion", "lane": "second",
           "reproduction": True, "original_seed": 20261007,
           "seeds": SEED_SETS["K2"], "ticks_per_phase": K2_TICKS,
           "per_seed": k2, "passes": n, "reproduces": ok, "verdict": line,
           "prereg": {"path": PREREG_PATH,
                      "receipt_hash": prereg["receipt_hash"],
                      "preregistered_utc": prereg["preregistered_utc"]},
           "prev_receipt_hash": chain[-1]}
    rec["receipt_hash"] = seal(rec)
    path = store("repro_second_lane_k2_broadcast_lesion.json", rec)
    chain.append(rec["receipt_hash"])
    print(f"K2: {line}\n  {path}")
    summary["K2"] = ok

    print("== second lane: K3 ignition probe ==")
    k3 = lane_k3()
    n, ok, line = verdict_line(k3, "pass")
    rec = {"experiment": "K3_ignition_linearity_probe", "lane": "second",
           "reproduction": True, "original_seed": 20261007,
           "seeds": SEED_SETS["K3"], "theta": K3_THETA, "points": K3_POINTS,
           "trials_per_point": K3_TRIALS, "jitter_sd": K3_JITTER,
           "per_seed": k3, "passes": n, "reproduces": ok, "verdict": line,
           "prereg": {"path": PREREG_PATH,
                      "receipt_hash": prereg["receipt_hash"],
                      "preregistered_utc": prereg["preregistered_utc"]},
           "prev_receipt_hash": chain[-1]}
    rec["receipt_hash"] = seal(rec)
    path = store("repro_second_lane_k3_ignition_probe.json", rec)
    chain.append(rec["receipt_hash"])
    print(f"K3: {line}\n  {path}")
    summary["K3"] = ok

    print("== second lane: K4 attention baseline ==")
    k4 = lane_k4()
    n, ok, line = verdict_line(k4, "win")
    rec = {"experiment": "K4_attention_baseline", "lane": "second",
           "reproduction": True, "original_seeds": [11, 22, 33, 44],
           "seeds": SEED_SETS["K4"], "ticks": K4_TICKS, "theta": 0.45,
           "margin": K4_MARGIN, "per_seed": k4, "wins": n,
           "reproduces": ok, "verdict": line,
           "prereg": {"path": PREREG_PATH,
                      "receipt_hash": prereg["receipt_hash"],
                      "preregistered_utc": prereg["preregistered_utc"]},
           "prev_receipt_hash": chain[-1]}
    rec["receipt_hash"] = seal(rec)
    path = store("repro_second_lane_k4_attention_baseline.json", rec)
    chain.append(rec["receipt_hash"])
    print(f"K4: {line}\n  {path}")
    summary["K4"] = ok

    print("\n=== SECOND-LANE SUMMARY ===")
    for k, v in summary.items():
        print(f"A-{k}: {'REPRODUCES' if v else 'FAILS TO REPRODUCE'}")
    return all(summary.values())


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
