"""Exploratory gain-landscape mapping (NOT the preregistered experiment).

Question: does ANY fixed gain vector beat frozen (1.0) by >= 1.30x on
canonical delayed_reward, and if so, which direction?
Uses exploratory seeds {501..506} — disjoint from any confirmatory seeds.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention import AttentionArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import CanonicalAdapter, neutral_specialists

CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
EXPLORE_SEEDS = (501, 502, 503, 504, 505, 506)


def run_fixed_gains(seed, gains):
    wk = WorkspaceTick(CHANNELS, neutral_specialists(CHANNELS, seed),
                       capacity=4, frozen_gains=True, gain_lr=0.15,
                       ignition_kwargs={"theta": 0.45},
                       arbitrator_cls=AttentionArbitrator)
    wk.arbitrator.gains = dict(gains)  # frozen -> pinned
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(CHANNELS)},
                           n_episodes=8)
    for _ in range(8 * DelayedReward.MAX_STEPS):
        wk.step(env.observe(), env)
    return sum(wk.rewards)


def main():
    vectors = {
        "frozen(1,1,1,1)": {"branch_a": 1.0, "branch_b": 1.0,
                            "forward": 1.0, "stay": 1.0},
        "fwd1.5": {"branch_a": 1.0, "branch_b": 1.0,
                   "forward": 1.5, "stay": 1.0},
        "fwd2.0": {"branch_a": 1.0, "branch_b": 1.0,
                   "forward": 2.0, "stay": 1.0},
        "fwd2_br0.5": {"branch_a": 0.5, "branch_b": 0.5,
                       "forward": 2.0, "stay": 1.0},
        "br2_fwd1": {"branch_a": 2.0, "branch_b": 2.0,
                     "forward": 1.0, "stay": 1.0},
        "br2_fwd2": {"branch_a": 2.0, "branch_b": 2.0,
                     "forward": 2.0, "stay": 1.0},
        "stay0.01": {"branch_a": 1.0, "branch_b": 1.0,
                     "forward": 1.0, "stay": 0.01},
        "fwd2_stay0.01": {"branch_a": 1.0, "branch_b": 1.0,
                          "forward": 2.0, "stay": 0.01},
        "br1.5_fwd2_stay0.01": {"branch_a": 1.5, "branch_b": 1.5,
                                "forward": 2.0, "stay": 0.01},
    }
    results = {}
    for name, gains in vectors.items():
        totals = [run_fixed_gains(s, gains) for s in EXPLORE_SEEDS]
        results[name] = totals
        print(f"{name:22s} " +
              " ".join(f"{t:5.2f}" for t in totals) +
              f"  mean={sum(totals)/len(totals):5.2f}")
    base = results["frozen(1,1,1,1)"]
    print("\nRatios vs frozen (per-seed mean):")
    for name, totals in results.items():
        if name.startswith("frozen"):
            continue
        ratios = [t / b if b > 0 else float("inf")
                  for t, b in zip(totals, base)]
        mean_r = sum(ratios) / len(ratios)
        nwins = sum(1 for r in ratios if r >= 1.30)
        print(f"{name:22s} meanR={mean_r:5.2f} wins(>=1.30)={nwins}/6 " +
              " ".join(f"{r:4.2f}" for r in ratios))


if __name__ == "__main__":
    main()
