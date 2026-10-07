"""K7 — BID ABLATION FOR CAUSAL BIDS (Phase 4 gap 3).

Bids sit at INTEGRATED: executed every tick, consumed by arbitration,
habituation measured — but no explicit lesion proves they do causal
work. This experiment lesions them.

Hypothesis: lesioning bids causally changes arbitration behavior —
  with bids replaced by a constant stub, the agent loses
  signal-tracking (the NR-A-005 condition where intact change-bids
  alone track relevance reversals).
Null: arbitration is unchanged by the bid lesion (bids are decorative;
  demote, do not promote).
Preregistered metric: on the signal-tracking reversal task
  (ChangingRelevanceEnv, stationary_signals=False, noise=0.05,
  theta=0.45 — the NR-A-005 condition), T = total(intact bids, frozen
  gains) / total(lesioned bids, frozen gains). PASS (promote bids to
  CAUSAL) requires T >= 1.30 on >= 3 of 4 fresh seeds {777, 888, 999,
  1212}. Frozen gains isolate the bid mechanism (no gain confound).
Baseline: intact bids + frozen gains (the NR-A-005 frozen baseline,
  which tracks the reversal through change-bids alone).
Ablation: experiment-side lesion — every channel's RunningZScoreBid is
  replaced by a constant 0.5 stub (no production-code change). With
  frozen gains at 1.0, arbitration collapses to the canonical tie-break
  (channel order), the direct "arbitration must change" evidence.
Procedure: 200 ticks/condition; conditions: (A) intact+frozen,
  (B) lesioned+frozen, (C) intact+learned (reference), (D)
  lesioned+learned (informative: do gains compensate for dead bids?).
  Winner-entropy under (A) vs (B) recorded as the direct arbitration-
  change measure.
Seed: 777, 888, 999, 1212 (fresh).
Kill: T < 1.30 on majority of seeds -> bids are not causal for
  arbitration -> keep INTEGRATED (or demote on evidence).
Interpretation/limitations: filled after the run.
"""
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists

SEEDS = (777, 888, 999, 1212)
TICKS = 200
THETA = 0.45
MARGIN = 1.30
NEED = 3


class ConstantBid:
    """Lesion stub: bids 0.5 regardless of stimulus. Replaces
    RunningZScoreBid experiment-side (no production change)."""

    def bid(self, value: float) -> float:
        return 0.5


def run(seed, lesion_bids, frozen):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    if lesion_bids:
        wk.arbitrator._bids = {c: ConstantBid() for c in channels}
    env = ChangingRelevanceEnv(channels, TICKS, seed=seed,
                               stationary_signals=False, noise=0.05)
    winners = []
    for _ in range(TICKS):
        tr = wk.step(env.observe(), env)
        winners.append(tr["arbitration"]["winner"])
    total = sum(wk.rewards)
    counts = Counter(winners)
    entropy = -sum((n / TICKS) * math.log2(n / TICKS) for n in counts.values())
    return {"total": round(total, 2),
            "winner_entropy_bits": round(entropy, 3),
            "winner_counts": dict(counts)}


def main():
    print("K7 — bid ablation: lesion bids -> arbitration must change")
    print(f"preregistered: T=total(A)/total(B) >= {MARGIN} on >= "
          f"{NEED}/{len(SEEDS)} seeds -> bids CAUSAL")
    results, wins = {}, 0
    for seed in SEEDS:
        a = run(seed, lesion_bids=False, frozen=True)
        b = run(seed, lesion_bids=True, frozen=True)
        c = run(seed, lesion_bids=False, frozen=False)
        d = run(seed, lesion_bids=True, frozen=False)
        t = a["total"] / b["total"] if b["total"] > 0 else float("inf")
        win = t >= MARGIN
        wins += int(win)
        results[str(seed)] = {
            "A_intact_frozen": a, "B_lesioned_frozen": b,
            "C_intact_learned": c, "D_lesioned_learned": d,
            "T": round(t, 3), "win": bool(win)}
        print(f"  seed={seed}: A={a['total']} B={b['total']} T={t:.2f} "
              f"{'WIN' if win else 'loss'} | winner_entropy A={a['winner_entropy_bits']} "
              f"B={b['winner_entropy_bits']} | C={c['total']} D={d['total']}")
    gate = wins >= NEED
    verdict = ("PASS — lesioning bids destroys signal-tracking "
               f"({wins}/{len(SEEDS)} seeds at T>={MARGIN}); bids are "
               "CAUSAL for arbitration" if gate else
               "FAIL — bid lesion does not change arbitration outcomes; "
               "bids stay INTEGRATED (not promoted)")
    print(f"wins: {wins}/{len(SEEDS)} (need {NEED})")
    print("VERDICT:", verdict)
    receipt = {"experiment": "K7_bid_ablation", "seeds": list(SEEDS),
               "ticks": TICKS, "theta": THETA, "margin": MARGIN,
               "need": NEED, "results": results, "wins": wins,
               "pass": bool(gate), "verdict": verdict,
               "lesion": ("experiment-side: arbitrator._bids replaced by "
                          "ConstantBid stubs returning 0.5; no production "
                          "code changed")}
    out = os.path.join(TREE, "receipts", "k7_bid_ablation.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
