"""EXP-FP-0080 ablations (preregistered plan; descriptive, not gated).

Fires because the primary gate PASSED. Four ablation arms on the SAME 4
seeds {82021..82024}, reusing the primary run's frozen totals (frozen is
variant-independent: frozen=True returns before any rate param matters).

Arms:
  (a) no_boost      -- corridor_boost = 0.0
  (b) no_prebranch  -- prebranch_demote = 0.0
  (c) no_punish     -- shape_punish = 0.0
  (d) branch_only   -- success locks in the branch choice but NOT the
                       progress channel (subclass override; module untouched)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from exp_fp_0080_ecr_repro import (  # noqa: E402
    probe as _probe, SEEDS, CHANNELS, THETA, GAIN_LR,
    ARBITRATOR_KWARGS, ReturnConditionedEpisodicArbitrator, MARGIN,
)
from tick import WorkspaceTick  # noqa: E402
from delayed_reward import DelayedReward  # noqa: E402
from k5_multitask_generalization import (  # noqa: E402
    CanonicalAdapter, neutral_specialists)


class BranchOnlyECR(ReturnConditionedEpisodicArbitrator):
    """ECR without the progress-channel lock-in on success."""

    def _absorb_success(self):
        branch_choice = self._ledger.first_branch(self.branch_channels)
        note = {}
        if branch_choice is not None:
            self.gains[branch_choice] = self.gain_cap
            note["branch_credit"] = branch_choice
        else:
            note["branch_credit"] = None
        note["progress_credit"] = "SKIPPED(branch_only)"
        self._ledger.clear()
        return note


def probe_variant(seed, cls, kwargs):
    wk = WorkspaceTick(CHANNELS, neutral_specialists(CHANNELS, seed),
                       capacity=4, frozen_gains=False, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=cls,
                       arbitrator_kwargs=dict(kwargs))
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(CHANNELS)},
                           n_episodes=8)
    n_success = 0
    for _ in range(8 * DelayedReward.MAX_STEPS):
        tr = wk.step(env.observe(), env)
        if tr.get("reward", 0.0) >= 1.0:
            n_success += 1
    return sum(wk.rewards), n_success


def main():
    with open(os.path.join(FLESH, "var",
                           "ecr-repro-0080-results.json")) as f:
        primary = json.load(f)
    frozen_total = {p["seed"]: p["frozen_total"]
                    for p in primary["summary"]["per_seed"]}
    primary_r = {p["seed"]: p["ratio"]
                 for p in primary["summary"]["per_seed"]}

    base = dict(ARBITRATOR_KWARGS)
    variants = {
        "no_boost": (ReturnConditionedEpisodicArbitrator,
                     {**base, "corridor_boost": 0.0}),
        "no_prebranch": (ReturnConditionedEpisodicArbitrator,
                         {**base, "prebranch_demote": 0.0}),
        "no_punish": (ReturnConditionedEpisodicArbitrator,
                      {**base, "shape_punish": 0.0}),
        "branch_only": (BranchOnlyECR, base),
    }
    out = {"experiment_id": "EXP-FP-0080-ABL",
           "note": "descriptive ablations, preregistered plan, not gated",
           "arms": {}}
    for name, (cls, kwargs) in variants.items():
        arm = {"seeds": {}}
        wins = 0
        for seed in SEEDS:
            tl, n_suc = probe_variant(seed, cls, kwargs)
            tf = frozen_total[seed]
            r = tl / tf if tf > 0 else float("inf")
            win = r >= MARGIN
            wins += int(win)
            arm["seeds"][str(seed)] = {
                "learned_total": tl, "frozen_total": tf,
                "ratio": r, "win": bool(win), "learned_successes": n_suc,
                "primary_ratio": primary_r[seed],
            }
            print(f"  {name:13s} seed={seed}: learned={tl:.2f} "
                  f"(succ={n_suc}) frozen={tf:.2f} R={r:.2f} "
                  f"{'WIN' if win else 'LOSS'} (primary R="
                  f"{primary_r[seed]:.2f})")
        arm["wins"] = wins
        out["arms"][name] = arm
        print(f"  {name:13s} -> {wins}/4 wins")
    path = os.path.join(FLESH, "var", "ecr-repro-0080-ablations.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("ablations:", path)


if __name__ == "__main__":
    main()
