"""K9 — SPARSE-REWARD GAIN REDESIGN (NR-A-006 candidate).

PREREGISTRATION (written 2026-10-07T03:55 EDT, BEFORE implementation/execution;
canonical copy: receipts/prereg_k9_sparse_reward_redesign.json):

  Hypothesis (H1): replacing the constant-baseline (0.5) delta rule with a
    return-conditioned, baseline-free eligibility-trace gain update lets
    learned gains beat frozen gains on canonical delayed_reward
    (R >= 1.30). Mechanism: (1) no baseline -> no universal gain decay
    under sparse reward (the NR-A-006 decay mechanism is removed by
    construction); (2) the eligibility trace carries the delayed +1.0 back
    to the t=0 branch choice (weight lambda^10 ~= 0.35), fixing the
    temporal credit-assignment failure.
  Null (H0): the redesign does not beat frozen gains (R < 1.30 on >= 2 of
    4 seeds) -> NR-A-006 stands as STRUCTURAL. Predicted failure
    mechanism (stated before running): trace credit concentrates on
    `forward` (most frequent pre-reward action); forward gain -> cap;
    forward wins the t=0 arbitration (a no-op there); episodes truncate;
    the baseline-free rule cannot recover -> learned ~= first lucky
    episode only. A state-blind bandit cannot learn branch-then-forward
    sequencing; the trace rule may only change HOW it fails.
  Metric: R = total(learned) / total(frozen) per seed, 8 episodes x <=15
    steps on canonical delayed_reward v1.0.0.
  Gate: BOUND LIFTED iff R >= 1.30 on >= 3 of 4 fresh seeds
    {1111, 2222, 3333, 4444}. Else BOUND STANDS -> NR-A-011.
  Config: identical to K5 P1 in EVERY hyperparameter except the update
    rule (channels, neutral specialists, capacity=4, gain_lr=0.15,
    theta=0.45, trace lambda=0.9, gain_cap=2.0, floor 0.01). Frozen =
    identical SparseRewardArbitrator with frozen=True (gains pinned 1.0).

Kill: R < 1.30 on majority of seeds -> the redesign fails -> record NR.
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention_sparse import SparseRewardArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)

SEEDS = (1111, 2222, 3333, 4444)
THETA = 0.45
MARGIN = 1.30
NEED = 3
TRACE_LAMBDA = 0.9
GAIN_LR = 0.15
EXPERIMENT_ID = "K9-sparse-reward-redesign"


def probe_delayed_reward_redesign(seed, frozen):
    """K5 P1 protocol, but the arbitrator is the K9 redesign."""
    channels = ["branch_a", "branch_b", "forward", "stay"]
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=SparseRewardArbitrator)
    assert isinstance(wk.arbitrator, SparseRewardArbitrator)
    assert abs(wk.arbitrator.trace_lambda - TRACE_LAMBDA) < 1e-12
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(channels)},
                           n_episodes=8)
    for _ in range(8 * DelayedReward.MAX_STEPS):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), dict(wk.arbitrator.gains),
            len(wk.arbitrator.gain_history))


def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def main():
    t_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []

    def emit(kind, payload):
        rec = {"experiment_id": EXPERIMENT_ID, "kind": kind, "t_wall": t_wall}
        rec.update(payload)
        rec["prev_hash"] = (records[-1]["receipt_hash"]
                            if records else "GENESIS")
        rec["receipt_hash"] = receipt_hash(rec)
        records.append(rec)
        return rec

    emit("preregistration", {
        "hypothesis": "H1: return-conditioned baseline-free eligibility-trace "
                      "gain update beats frozen gains on canonical "
                      "delayed_reward (R >= 1.30, >=3/4 seeds)",
        "null": "H0: no beat (R < 1.30 majority) -> NR-A-006 stands "
                "STRUCTURAL",
        "metric": "R = total(learned)/total(frozen) per seed",
        "gate": f"BOUND LIFTED iff R >= {MARGIN} on >= {NEED}/{len(SEEDS)} "
                f"fresh seeds {list(SEEDS)}",
        "config": {"env": "delayed_reward v1.0.0", "episodes": 8,
                   "max_steps": DelayedReward.MAX_STEPS,
                   "channels": ["branch_a", "branch_b", "forward", "stay"],
                   "specialists": "neutral (K5 P1 identical)",
                   "capacity": 4, "gain_lr": GAIN_LR, "theta": THETA,
                   "trace_lambda": TRACE_LAMBDA, "gain_cap": 2.0,
                   "floor": 0.01, "arbitrator": "SparseRewardArbitrator",
                   "frozen": "same class, frozen=True, gains pinned 1.0"},
        "prereg_file": "receipts/prereg_k9_sparse_reward_redesign.json",
    })

    print(f"{EXPERIMENT_ID}: R=learned/frozen >= {MARGIN} on >= "
          f"{NEED}/{len(SEEDS)} seeds lifts the NR-A-006 bound")
    wins = 0
    for seed in SEEDS:
        tl, gains_l, n_upd = probe_delayed_reward_redesign(seed, False)
        tf, gains_f, _ = probe_delayed_reward_redesign(seed, True)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        emit("seed_result", {
            "seed": seed,
            "learned_total": round(tl, 3), "frozen_total": round(tf, 3),
            "ratio": round(r, 3), "win": bool(win),
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
            "n_gain_updates": n_upd,
        })
        print(f"  seed={seed}: learned={tl:.2f} frozen={tf:.2f} R={r:.2f} "
              f"{'WIN' if win else 'loss'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    lifted = wins >= NEED
    emit("verdict", {"wins": wins, "need": NEED, "margin": MARGIN,
                     "bound_lifted": bool(lifted),
                     "interpretation": (
                         "BOUND LIFTED: trace rule beats frozen on "
                         "delayed_reward" if lifted else
                         "BOUND STANDS (NR-A-006 structural): redesign "
                         "does not beat frozen; record NR-A-011")})
    print(f"  -> {'BOUND LIFTED' if lifted else 'BOUND STANDS'} "
          f"({wins}/{len(SEEDS)})")

    out = os.path.join(TREE, "receipts", "k9_sparse_reward_redesign.ndjson")
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print("receipt:", out)


if __name__ == "__main__":
    main()
