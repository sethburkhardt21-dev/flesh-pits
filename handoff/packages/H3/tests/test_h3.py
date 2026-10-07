"""H3 package test — recurrent ignition gate + R1 sub-ignition path.

Standalone proof from the packaged sources:
  K3 (ignition probe): sweep salience 0..1 through RecurrentIgnition;
    the real gate must be a Heaviside step (transition width W < 0.12,
    P=0 below theta-0.25, P=1 above theta+0.25) while the graded control
    reads linear (probe is not blind).
  R1 (sub-ignition exploration, wired default in tick.py): on ticks where
    nothing ignites, the tick still acts (no freeze) via the graded
    arbitration winner, with ZERO non-feedback consumer deliveries on
    those trials (the gate's selectivity is untouched by exploration).

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only. Seeded.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ignition import RecurrentIgnition
from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists

SEED = 20261007
N_POINTS = 51
TRIALS = 200
JITTER_SD = 0.03
THETA = 0.6
R1_SEEDS = (51501, 51502, 51503)
R1_TICKS = 200
R1_THETA = 0.6  # K8 regime: theta=0.6 + frozen gains -> phase-2 freeze


def sweep(linear_probe, seed):
    rng = random.Random(seed)
    curve = []
    for i in range(N_POINTS):
        x = i / (N_POINTS - 1)
        hits = 0
        for _ in range(TRIALS):
            ig = RecurrentIgnition(theta=THETA, linear_probe=linear_probe)
            rec = ig.ignite(x, jitter=rng.gauss(0, JITTER_SD))
            p = rec["strength"] if linear_probe else (
                1.0 if rec["ignited"] else 0.0)
            hits += p
        curve.append((x, hits / TRIALS))
    return curve


def transition_width(curve, lo=0.05, hi=0.95):
    try:
        x_lo = next(x for x, p in curve if p >= lo)
        x_hi = next(x for x, p in curve if p >= hi)
    except StopIteration:
        return float("inf")
    return x_hi - x_lo


def run_k3():
    real = sweep(False, SEED)
    graded = sweep(True, SEED + 1)
    w = transition_width(real)
    p_low = (sum(p for x, p in real if x <= THETA - 0.25)
             / max(1, sum(1 for x, p in real if x <= THETA - 0.25)))
    p_high = (sum(p for x, p in real if x >= THETA + 0.25)
              / max(1, sum(1 for x, p in real if x >= THETA + 0.25)))
    lin_err = sum(abs(p - x) for x, p in graded) / len(graded)
    print(f"  real gate: W={w:.3f} (need < 0.12); "
          f"P(x<=0.35)={p_low:.3f} (need 0); P(x>=0.85)={p_high:.3f} "
          f"(need 1); graded linearity err={lin_err:.3f} (need < 0.08)")
    gate = (w < 0.12 and p_low == 0.0 and p_high == 1.0 and lin_err < 0.08)
    print("  K3 ->", "PASS" if gate else "FAIL")
    return gate, {"transition_width": w, "p_low": p_low, "p_high": p_high,
                  "graded_linearity_error": lin_err}


class HoldTick(WorkspaceTick):
    """Pre-K8 behavior: with no planner proposal (nothing ignited), hold
    the last action. This is the freeze the R1 path was wired to remove."""

    def _select_action(self, proposal, decision):
        if proposal and proposal.get("proposed_action"):
            return proposal["proposed_action"], {"path": "ignited_proposal"}
        return self._last_action, {"path": "hold"}


def run_r1():
    # K8 regime: learned gains + theta=0.6 -> phase-2 freeze under the
    # pre-R1 hold branch; the wired R1 path must recover action selection
    # (phase-2 'c' fraction >> 0) with zero non-feedback consumer
    # deliveries on sub-ignition trials.
    channels = ["a", "b", "c"]
    per_seed, all_ok = {}, True
    for seed in R1_SEEDS:
        row = {}
        for name, cls in (("hold", HoldTick), ("r1", WorkspaceTick)):
            wk = cls(channels, make_specialists(channels), capacity=3,
                     frozen_gains=False, gain_lr=0.15,
                     ignition_kwargs={"theta": R1_THETA})
            env = ChangingRelevanceEnv(channels, R1_TICKS, seed=seed,
                                       stationary_signals=True)
            sub_ign_ticks = []
            for _ in range(R1_TICKS):
                tr = wk.step(env.observe(), env)
                if tr.get("action_selection", {}).get("path") \
                        == "sub_ignition_explore":
                    sub_ign_ticks.append(tr["tick"])
            violations = []
            for t in sub_ign_ticks:
                for d in wk.bus.deliveries:
                    if d["tick"] != t or d["status"] != "delivered":
                        continue
                    if d["kind"] != "feedback" \
                            or d["consumer"] != "attention_update":
                        violations.append({"tick": t, "kind": d["kind"],
                                           "consumer": d["consumer"]})
            p2 = wk.actions[R1_TICKS // 2:]
            c_frac = sum(1 for a in p2 if a == "c") / len(p2)
            row[name] = {"phase2_c_fraction": round(c_frac, 3),
                         "sub_ignition_trials": len(sub_ign_ticks),
                         "violations": len(violations),
                         "actions": len(wk.actions)}
        ok = (row["hold"]["phase2_c_fraction"] < 0.10
              and row["r1"]["phase2_c_fraction"] > 0.50
              and row["r1"]["sub_ignition_trials"] > 0
              and row["r1"]["violations"] == 0
              and row["r1"]["actions"] == R1_TICKS)
        all_ok = all_ok and ok
        row["ok"] = ok
        per_seed[str(seed)] = row
        print(f"  seed={seed}: hold phase2_c={row['hold']['phase2_c_fraction']:.2f} "
              f"r1 phase2_c={row['r1']['phase2_c_fraction']:.2f} "
              f"r1 sub-ign={row['r1']['sub_ignition_trials']} "
              f"violations={row['r1']['violations']} "
              f"-> {'ok' if ok else 'FAIL'}")
    print("  R1 ->", "PASS" if all_ok else "FAIL")
    return all_ok, per_seed


def main():
    print("K3 — ignition linearity probe")
    g3, r3 = run_k3()
    print("R1 — sub-ignition exploration keeps the gate selective")
    gr, rr = run_r1()
    gate = g3 and gr
    print("OVERALL ->", "PASS" if gate else "FAIL")
    receipt = {"package": "H3", "test": "test_h3.py",
               "K3": {"pass": bool(g3), "results": r3},
               "R1": {"pass": bool(gr), "results": rr},
               "pass": bool(gate),
               "verdict": "PASS" if gate else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
