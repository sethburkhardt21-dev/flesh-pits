"""EXP-FP-0021 -- EPISODIC CONTRASTIVE RETURN (ECR) GAIN REDESIGN.

PREREGISTRATION (sealed 2026-10-07T08:45 EDT, BEFORE any execution on the
confirmatory seeds; canonical copy:
flesh-pits/experiments/preregistration_EXP-FP-0021.json):

  Hypothesis (H1): replacing the constant-baseline delta rule (and K9's
    eligibility-trace rule) with the Episodic Contrastive Return (ECR)
    gain update lets learned gains beat frozen gains on canonical
    delayed_reward (R >= 1.30 on 4/4 fresh seeds).
  Null (H0): ECR does not beat frozen (R < 1.30 on >= 1 seed).
  Metric: R = total(learned)/total(frozen) per seed, 8 episodes x <=15
    steps on canonical delayed_reward v1.0.0.
  Gate: PASS iff R >= 1.30 on 4/4 fresh seeds {21001..21004}.
  Config: identical to K5 P1 in EVERY hyperparameter except the update
    rule. Frozen = identical EpisodicContrastiveArbitrator with
    frozen=True (gains pinned at 1.0).

Kill: R < 1.30 on any seed -> FAIL -> negative result with mechanism.
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)  # prototypes/architecture-a
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention_ecr import EpisodicContrastiveArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)

SEEDS = (21001, 21002, 21003, 21004)
THETA = 0.45
MARGIN = 1.30
GAIN_LR = 0.15
EXPERIMENT_ID = "EXP-FP-0021"
CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
ARBITRATOR_KWARGS = {
    "branch_channels": {"branch_a", "branch_b"},
    "max_steps": DelayedReward.MAX_STEPS,
    "shape_punish": 0.02,
    "prebranch_demote": 0.05,
    "corridor_boost": 0.01,
}


def probe(seed, frozen):
    """K5 P1 protocol with the ECR arbitrator."""
    wk = WorkspaceTick(CHANNELS, neutral_specialists(CHANNELS, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=EpisodicContrastiveArbitrator,
                       arbitrator_kwargs=dict(ARBITRATOR_KWARGS))
    assert isinstance(wk.arbitrator, EpisodicContrastiveArbitrator)
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
        "hypothesis": "H1: Episodic Contrastive Return (ECR) gain update "
                      "beats frozen gains on canonical delayed_reward "
                      "(R >= 1.30, 4/4 fresh seeds)",
        "null": "H0: R < 1.30 on >= 1 seed -> redesign fails",
        "metric": "R = total(learned)/total(frozen) per seed",
        "gate": f"PASS iff R >= {MARGIN} on 4/4 fresh seeds {list(SEEDS)}",
        "config": {"env": "delayed_reward v1.0.0", "episodes": 8,
                   "max_steps": DelayedReward.MAX_STEPS,
                   "channels": CHANNELS,
                   "specialists": "neutral (K5 P1 identical)",
                   "capacity": 4, "gain_lr": GAIN_LR, "theta": THETA,
                   "gain_cap": 2.0, "floor": 0.01,
                   "arbitrator": "EpisodicContrastiveArbitrator",
                   "arbitrator_params": {
                       k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in ARBITRATOR_KWARGS.items()},
                   "frozen": "same class, frozen=True, gains pinned 1.0"},
        "prereg_file": "experiments/preregistration_EXP-FP-0021.json",
    })

    print(f"{EXPERIMENT_ID}: R=learned/frozen >= {MARGIN} on 4/4 seeds "
          f"lifts the NR-A-006 bound")
    wins = 0
    for seed in SEEDS:
        tl, gains_l, n_upd, n_suc_l = probe(seed, False)
        tf, gains_f, _, n_suc_f = probe(seed, True)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        emit("seed_result", {
            "seed": seed,
            "learned_total": round(tl, 3), "frozen_total": round(tf, 3),
            "ratio": round(r, 3), "win": bool(win),
            "learned_successes": n_suc_l, "frozen_successes": n_suc_f,
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
            "n_gain_updates": n_upd,
        })
        print(f"  seed={seed}: learned={tl:.2f} (succ={n_suc_l}) "
              f"frozen={tf:.2f} (succ={n_suc_f}) R={r:.2f} "
              f"{'WIN' if win else 'LOSS'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    passed = (wins == len(SEEDS))
    emit("verdict", {"wins": wins, "need": len(SEEDS), "margin": MARGIN,
                     "passed": bool(passed),
                     "interpretation": (
                         "PASS: ECR lifts the NR-A-006 bound on "
                         "delayed_reward (4/4 seeds)" if passed else
                         "FAIL: ECR does not lift the bound; record "
                         "negative result with mechanism")})
    print(f"  -> {'PASS' if passed else 'FAIL'} ({wins}/{len(SEEDS)})")

    out = os.path.join(FLESH, "receipts", "EXP-FP-0021-ecr.ndjson")
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print("receipt:", out)

    # G3: verify the hash chain we just wrote.
    prev = "GENESIS"
    with open(out) as f:
        for line in f:
            rec = json.loads(line)
            assert rec["prev_hash"] == prev, "chain break"
            body = {k: v for k, v in rec.items() if k != "receipt_hash"}
            assert (hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest()
                    == rec["receipt_hash"]), "hash mismatch"
            prev = rec["receipt_hash"]
    print(f"G3: hash chain verified ({len(records)} records)")


if __name__ == "__main__":
    main()
