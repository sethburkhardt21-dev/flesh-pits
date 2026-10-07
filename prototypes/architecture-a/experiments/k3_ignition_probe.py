"""K3 — IGNITION LINEARITY PROBE (Architecture A load-bearing claim 2).

Hypothesis: ignition is a nonlinear threshold event. Sweeping input
  salience, propagation probability must follow a sigmoid/Heaviside
  STEP: zero below threshold, one above, with a narrow transition —
  NOT a graded linear leak.
Null: propagation probability is linear in salience (weighted
  averaging, no ignition event).
Preregistered metric: transition width W = x(P=0.95) - x(P=0.05)
  measured on the probability curve. PASS requires W < 0.12 AND
  P(x <= theta-0.25) == 0 AND P(x >= theta+0.25) == 1.
  The graded control (linear passthrough) must show |P(x) - x| small
  (i.e. the probe CAN detect linearity — it is not blind to it).
Baseline/ablation: linear_probe=True ignition (graded passthrough,
  no gate, no recurrence) run through the identical sweep.
Procedure: x in [0, 1] step 0.02 (51 points); 200 trials/point; per
  trial jitter ~ N(0, 0.03) from a seeded RNG (fresh RecurrentIgnition
  per trial so reverberant state cannot smear the curve; strength
  reset each trial). Propagation = ignited flag (the hard gate).
Seed: 20261007.
Kill: linear graded propagation under the REAL gate -> there is no
  ignition event, only weighted averaging; rename honestly ("a queue
  with weights") and demote the module.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ignition import RecurrentIgnition

SEED = 20261007
N_POINTS = 51
TRIALS = 200
JITTER_SD = 0.03
THETA = 0.6


def sweep(linear_probe, seed):
    rng = random.Random(seed)
    curve = []
    for i in range(N_POINTS):
        x = i / (N_POINTS - 1)
        hits = 0
        for _ in range(TRIALS):
            ig = RecurrentIgnition(theta=THETA, linear_probe=linear_probe)
            rec = ig.ignite(x, jitter=rng.gauss(0, JITTER_SD))
            # graded control: propagation strength IS the signal
            p = rec["strength"] if linear_probe else (1.0 if rec["ignited"] else 0.0)
            hits += p
        curve.append((x, hits / TRIALS))
    return curve


def transition_width(curve, lo=0.05, hi=0.95):
    xs = [x for x, p in curve]
    try:
        x_lo = next(x for x, p in curve if p >= lo)
        x_hi = next(x for x, p in curve if p >= hi)
    except StopIteration:
        return float("inf")
    return x_hi - x_lo


def main():
    print("K3 — ignition linearity probe: sweep salience, measure P(propagate)")
    real = sweep(False, SEED)
    graded = sweep(True, SEED + 1)

    w = transition_width(real)
    p_low = sum(p for x, p in real if x <= THETA - 0.25) / \
        max(1, sum(1 for x, p in real if x <= THETA - 0.25))
    p_high = sum(p for x, p in real if x >= THETA + 0.25) / \
        max(1, sum(1 for x, p in real if x >= THETA + 0.25))
    lin_err = sum(abs(p - x) for x, p in graded) / len(graded)

    print(f"  real gate: transition width W={w:.3f} (need < 0.12)")
    print(f"  real gate: mean P(x<=0.35)={p_low:.3f} (need 0), "
          f"mean P(x>=0.85)={p_high:.3f} (need 1)")
    print(f"  graded control: mean|P(x)-x|={lin_err:.3f} (small => probe "
          f"detects linearity)")
    # show the step
    band = [(x, p) for x, p in real if 0.45 <= x <= 0.75]
    print("  curve near threshold:", [(round(x, 2), round(p, 2)) for x, p in band])

    gate = (w < 0.12 and p_low == 0.0 and p_high == 1.0 and lin_err < 0.08)
    verdict = ("PASS — propagation is a Heaviside step (W=%.3f); graded "
               "control is linear as expected; the gate is a real "
               "nonlinear event" % w if gate else
               "FAIL — kill condition met: propagation is graded/linear; "
               "there is no ignition event — rename honestly and demote")
    print("VERDICT:", verdict)
    receipt = {"experiment": "K3_ignition_linearity_probe", "seed": SEED,
               "theta": THETA, "points": N_POINTS, "trials": TRIALS,
               "jitter_sd": JITTER_SD, "transition_width": w,
               "p_below": p_low, "p_above": p_high,
               "graded_control_linearity_error": lin_err,
               "curve": [[round(x, 4), round(p, 4)] for x, p in real],
               "pass": bool(gate), "verdict": verdict}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "receipts", "k3_ignition_probe.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
