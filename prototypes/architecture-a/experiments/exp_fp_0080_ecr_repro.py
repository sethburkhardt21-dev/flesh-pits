"""EXP-FP-0080 -- INDEPENDENT REPLICATION of the ECR sparse-reward gain redesign.

Preregistration sealed 2026-10-07T09:17 EDT BEFORE the implementation was
written and BEFORE any run (canonical copy:
flesh-pits/experiments/preregistration_EXP-FP-0080.json).

Protocol: K5 P1 identical in EVERY hyperparameter except the update rule,
which is an INDEPENDENT implementation of the ECR specification
(ReturnConditionedEpisodicArbitrator in attention_ecr_repro.py; no code
shared with attention_ecr.py beyond the AttentionArbitrator base class).

  Metric: R = total(learned)/total(frozen) per seed, 8 episodes x <=15
  steps on canonical delayed_reward v1.0.0.
  Gate: REPRODUCED iff R >= 1.30 on 4/4 fresh seeds {82021..82024}.
  Kill: R < 1.30 on any seed -> FAIL -> negative result with mechanism.

Receipts: a second, harness-only script (write_receipt_0080.py) reads the
staged results and calls experiments/harness.py write_receipt
(hash-chained). The split keeps the architecture-a and experiments/envs
import domains from colliding on the module name `envs`. If the harness
refuses the write (fail-closed ReceiptExistsError), claim a fresh ID --
never force.
"""
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)  # prototypes/architecture-a
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention_ecr_repro import ReturnConditionedEpisodicArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)
from env_interface import config_hash, make_episode_id

SEEDS = (82021, 82022, 82023, 82024)
THETA = 0.45
MARGIN = 1.30
GAIN_LR = 0.15
EXPERIMENT_ID = "EXP-FP-0080"
CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
ARBITRATOR_KWARGS = {
    "branch_channels": {"branch_a", "branch_b"},
    "max_steps": DelayedReward.MAX_STEPS,
    "shape_punish": 0.02,
    "prebranch_demote": 0.05,
    "corridor_boost": 0.01,
}


def probe(seed, frozen):
    """K5 P1 protocol with the independent ECR arbitrator."""
    wk = WorkspaceTick(CHANNELS, neutral_specialists(CHANNELS, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=ReturnConditionedEpisodicArbitrator,
                       arbitrator_kwargs=dict(ARBITRATOR_KWARGS))
    assert isinstance(wk.arbitrator, ReturnConditionedEpisodicArbitrator)
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(CHANNELS)},
                           n_episodes=8)
    n_success = 0
    for _ in range(8 * DelayedReward.MAX_STEPS):
        tr = wk.step(env.observe(), env)
        if tr.get("reward", 0.0) >= 1.0:
            n_success += 1
    return (sum(wk.rewards), dict(wk.arbitrator.gains),
            len(wk.arbitrator.gain_history), n_success)


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"{EXPERIMENT_ID}: independent ECR replication; "
          f"R=learned/frozen >= {MARGIN} on 4/4 fresh seeds {list(SEEDS)}")

    episodes = []
    per_seed = []
    wins = 0
    for seed in SEEDS:
        tl, gains_l, n_upd, n_suc_l = probe(seed, False)
        tf, gains_f, _, n_suc_f = probe(seed, True)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        per_seed.append({
            "seed": seed,
            "learned_total": tl, "frozen_total": tf,
            "ratio": r, "win": bool(win),
            "learned_successes": n_suc_l, "frozen_successes": n_suc_f,
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
            "n_gain_updates": n_upd,
        })
        for arm, total, n_suc in (("learned", tl, n_suc_l),
                                  ("frozen", tf, n_suc_f)):
            episodes.append({
                "episode_id": make_episode_id(
                    DelayedReward.NAME, DelayedReward.VERSION, seed,
                    0 if arm == "learned" else 1),
                "seed": seed, "arm": arm, "n_env_episodes": 8,
                "max_steps": DelayedReward.MAX_STEPS,
                "return": total, "successes": n_suc,
                "truncated": False, "done": True,
            })
        print(f"  seed={seed}: learned={tl:.2f} (succ={n_suc_l}) "
              f"frozen={tf:.2f} (succ={n_suc_f}) R={r:.2f} "
              f"{'WIN' if win else 'LOSS'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    passed = (wins == len(SEEDS))
    print(f"  -> {'REPRODUCED' if passed else 'FAIL'} "
          f"({wins}/{len(SEEDS)})")

    returns = [e["return"] for e in episodes]
    config = {
        "env": "delayed_reward v1.0.0", "episodes_per_arm": 8,
        "max_steps": DelayedReward.MAX_STEPS, "channels": CHANNELS,
        "specialists": "neutral (K5 P1 identical)", "capacity": 4,
        "gain_lr": GAIN_LR, "theta": THETA, "gain_cap": 2.0, "floor": 0.01,
        "arbitrator": "ReturnConditionedEpisodicArbitrator "
                      "(independent ECR reimplementation)",
        "arbitrator_params": {
            k: (sorted(v) if isinstance(v, set) else v)
            for k, v in ARBITRATOR_KWARGS.items()},
        "frozen": "same class, frozen=True, gains pinned 1.0",
        "seeds": list(SEEDS), "replicates": "EXP-FP-0021",
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": SEEDS[0],
        "started_utc": started,
        "episodes": episodes,
        "summary": {
            "n_episodes": len(episodes),
            "mean_return": statistics.fmean(returns),
            "stdev_return": statistics.pstdev(returns)
            if len(returns) > 1 else 0.0,
            "min_return": min(returns), "max_return": max(returns),
            "mean_steps": 8 * DelayedReward.MAX_STEPS / 2,
            "per_seed": per_seed,
            "wins": wins, "need": len(SEEDS), "margin": MARGIN,
            "passed": bool(passed),
            "verdict": ("REPRODUCED" if passed else "FAIL"),
        },
    }

    # Stage the harness-shaped result for the receipt writer (a separate
    # harness-only process; keeps the architecture-a and experiments/envs
    # import domains from colliding on the module name `envs`).
    results_path = os.path.join(FLESH, "var", "ecr-repro-0080-results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("staged results:", results_path)

    # G2 determinism spot-check: recompute seed 82021's learned arm.
    tl2, _, _, _ = probe(SEEDS[0], False)
    tl1 = per_seed[0]["learned_total"]
    assert abs(tl2 - tl1) < 1e-12, f"G2 FAIL: {tl2} != {tl1}"
    print(f"G2: determinism spot-check PASS (seed {SEEDS[0]} "
          f"learned={tl1:.6f} exact)")


if __name__ == "__main__":
    main()
