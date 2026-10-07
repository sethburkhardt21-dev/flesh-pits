"""Exploratory ECR test (NOT preregistered). Learned (ECR) vs frozen on
exploratory seeds {501..506}. Tunes nothing on confirmatory seeds."""
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
from attention_ecr import EpisodicContrastiveArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import CanonicalAdapter, neutral_specialists

CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
SEEDS = (501, 502, 503, 504, 505, 506)


def run(seed, frozen):
    wk = WorkspaceTick(CHANNELS, neutral_specialists(CHANNELS, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": 0.45},
                       arbitrator_cls=EpisodicContrastiveArbitrator,
                       arbitrator_kwargs={
                           "branch_channels": {"branch_a", "branch_b"},
                           "max_steps": DelayedReward.MAX_STEPS})
    a = wk.arbitrator
    assert isinstance(a, EpisodicContrastiveArbitrator)
    assert a.branch_channels == {"branch_a", "branch_b"}
    assert a.max_steps == DelayedReward.MAX_STEPS
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(CHANNELS)},
                           n_episodes=8)
    n_success = 0
    for _ in range(8 * DelayedReward.MAX_STEPS):
        tr = wk.step(env.observe(), env)
        if tr.get("reward", 0.0) >= 1.0:
            n_success += 1
    return sum(wk.rewards), dict(wk.arbitrator.gains), n_success


def main():
    wins = 0
    for s in SEEDS:
        tl, gl, nl = run(s, False)
        tf, gf, nf = run(s, True)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= 1.30
        wins += win
        print(f"seed={s}: learned={tl:5.2f} (succ={nl}) "
              f"frozen={tf:5.2f} (succ={nf}) R={r:5.2f} "
              f"{'WIN' if win else 'loss'} gains={gl}")
    print(f"wins: {wins}/{len(SEEDS)}")


if __name__ == "__main__":
    main()
