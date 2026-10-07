"""H1 package test — A's learned-gain attention loop.

Standalone proof of the mechanism from the packaged sources:
  1. K4 procedure (preregistered): learned gains vs frozen gains on the
     canonical changing-relevance env (stationary signals, reward weights
     reverse at tick 100/200). PASS requires R = total(learned)/total(frozen)
     >= 1.30 on >= 3 of 4 seeds {11,22,33,44}.
  2. Identity symmetry: two novel channels with identical observations bid
     identically (label-agnostic tie-break; no identity-privileged machinery).

Exit 0 on PASS, 1 on FAIL. Writes ./receipts/package_test_receipt.json.
Stdlib only. Deterministic (seeded).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists
from attention import AttentionArbitrator

SEEDS = (11, 22, 33, 44)
TICKS = 200
THETA = 0.45
GAIN_LR = 0.15
MARGIN = 1.30
NEED = 3


def run_k4(frozen, seed):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA})
    env = ChangingRelevanceEnv(channels, TICKS, seed=seed,
                               stationary_signals=True)
    for _ in range(TICKS):
        wk.step(env.observe(), env)
    return sum(wk.rewards)


def identity_symmetry():
    """Two novel channels, identical observations -> identical bid paths."""
    arb = AttentionArbitrator(["zz_novel_a", "zz_novel_b"])
    traj_a, traj_b = [], []
    for i in range(50):
        x = 0.5 + 0.01 * ((i * 37) % 7)  # deterministic pseudo-signal
        da = arb.arbitrate({"zz_novel_a": x, "zz_novel_b": x})
        traj_a.append(da["competed_bids"]["zz_novel_a"])
        traj_b.append(da["competed_bids"]["zz_novel_b"])
    # bit-identical competed bids for the two novel labels
    return all(a == b for a, b in zip(traj_a, traj_b))


def main():
    results, wins = {}, 0
    for seed in SEEDS:
        tl = run_k4(False, seed)
        tf = run_k4(True, seed)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        results[str(seed)] = {"learned": round(tl, 2),
                              "frozen": round(tf, 2),
                              "ratio": round(r, 3), "win": win}
        print(f"  seed={seed}: learned={tl:.1f} frozen={tf:.1f} "
              f"R={r:.2f} {'WIN' if win else 'loss'}")
    sym = identity_symmetry()
    print("  identity symmetry (novel labels bid identically):", sym)
    gate = wins >= NEED and sym
    verdict = ("PASS" if gate else "FAIL")
    print(f"K4 wins: {wins}/{len(SEEDS)} (need {NEED}); symmetry={sym} -> "
          f"{verdict}")
    receipt = {"package": "H1", "test": "test_h1.py",
               "seeds": list(SEEDS), "ticks": TICKS, "theta": THETA,
               "margin": MARGIN, "need": NEED, "results": results,
               "wins": wins, "identity_symmetry": sym,
               "pass": bool(gate), "verdict": verdict}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
