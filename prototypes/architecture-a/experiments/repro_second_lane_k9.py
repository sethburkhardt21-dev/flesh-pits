"""REPRO-SECOND-LANE-K9 — independent replication of K9.

Spec: PREREGISTRATION receipts/prereg_repro_second_lane_k9.json
(sealed 2026-10-07T07:58:00Z, BEFORE this driver was written).

Protocol is the K9 spec from the prereg: canonical delayed_reward v1.0.0,
8 episodes x <=15 steps, SparseRewardArbitrator, neutral specialists,
capacity=4, gain_lr=0.15, theta=0.45, trace_lambda=0.9, gain_cap=2.0.
Metric: R = total(learned)/total(frozen) per seed; WIN iff R >= 1.30.
Fresh seeds {91511, 91512, 91513, 91514} (disjoint from original
{1111, 2222, 3333, 4444}).

Replication verdict: REPRODUCES iff wins <= 1/4 (BOUND STANDS again);
OVERTURNS iff wins >= 3/4; 2/4 = INCONCLUSIVE.

Imports only the architecture modules under test. Does NOT import the
original k9_sparse_reward_redesign.py script.
"""
import datetime
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TREE = os.path.dirname(_HERE)
_FLESH = os.path.dirname(os.path.dirname(_TREE))
sys.path.insert(0, os.path.join(_FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(_FLESH, "experiments"))
sys.path.insert(0, _TREE)
sys.path.insert(0, _HERE)

from tick import WorkspaceTick
from attention_sparse import SparseRewardArbitrator
from delayed_reward import DelayedReward
from k5_multitask_generalization import CanonicalAdapter, neutral_specialists

EXPERIMENT_ID = "REPRO-SECOND-LANE-K9"
TARGET_ID = "K9-sparse-reward-redesign"
SEEDS = (91511, 91512, 91513, 91514)
CHANNELS = ("branch_a", "branch_b", "forward", "stay")
EPISODES = 8
MARGIN = 1.30
NEED_STAND = 3      # wins <= NEED_STAND-2 (i.e. <=1/4) reproduces bound-stands


def chained_digest(record):
    body = {k: v for k, v in record.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


class ChainWriter:
    """Hash-chained ndjson ledger: each record embeds prev_hash."""

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.chain = []

    def log(self, kind, payload):
        rec = {"experiment_id": self.experiment_id, "kind": kind}
        rec.update(payload)
        rec["prev_hash"] = (self.chain[-1]["receipt_hash"]
                            if self.chain else "GENESIS")
        rec["receipt_hash"] = chained_digest(rec)
        self.chain.append(rec)
        return rec

    def dump(self, path):
        with open(path, "w") as f:
            for rec in self.chain:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        return path


def replicate_condition(seed, frozen):
    """One K9 condition (learned or frozen) on one seed."""
    wk = WorkspaceTick(
        list(CHANNELS), neutral_specialists(list(CHANNELS), seed),
        capacity=4, frozen_gains=frozen, gain_lr=0.15,
        ignition_kwargs={"theta": 0.45},
        arbitrator_cls=SparseRewardArbitrator)
    assert isinstance(wk.arbitrator, SparseRewardArbitrator), (
        "protocol requires the sparse-reward redesign arbitrator")
    assert abs(wk.arbitrator.trace_lambda - 0.9) < 1e-12, (
        "protocol requires the default trace_lambda=0.9")
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(CHANNELS)},
                           n_episodes=EPISODES)
    total_ticks = EPISODES * DelayedReward.MAX_STEPS
    for _ in range(total_ticks):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), {c: wk.arbitrator.gains[c] for c in CHANNELS},
            len(wk.arbitrator.gain_history))


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    t_wall = t0.isoformat()
    ledger = ChainWriter(EXPERIMENT_ID)

    ledger.log("preregistration", {
        "t_wall": t_wall,
        "target": TARGET_ID,
        "prereg_file": "receipts/prereg_repro_second_lane_k9.json",
        "seal_sha256": "237fd706e356bb9c452710eedfb35287c9f6352f2579fbc461bd63e02d72b7e4",
        "note": ("independent second-lane driver; preregistration sealed "
                 "before this driver was written"),
        "hypothesis": ("K9's BOUND STANDS (NR-A-011) verdict replicates on "
                       "fresh seeds"),
        "gate": f"WIN iff R >= {MARGIN}; BOUND STANDS iff wins <= 1/{len(SEEDS)}",
        "seeds": list(SEEDS),
    })

    wins = 0
    for seed in SEEDS:
        learned_total, learned_gains, n_updates = replicate_condition(
            seed, frozen=False)
        frozen_total, _, _ = replicate_condition(seed, frozen=True)
        ratio = (learned_total / frozen_total
                 if frozen_total > 0 else float("inf"))
        won = ratio >= MARGIN
        wins += int(won)
        ledger.log("seed_result", {
            "t_wall": t_wall,
            "target": TARGET_ID,
            "seed": seed,
            "learned_total": round(learned_total, 3),
            "frozen_total": round(frozen_total, 3),
            "ratio": round(ratio, 3),
            "win": bool(won),
            "learned_gains": {k: round(v, 3)
                              for k, v in learned_gains.items()},
            "n_gain_updates": n_updates,
        })
        print(f"  seed={seed} learned={learned_total:.2f} "
              f"frozen={frozen_total:.2f} R={ratio:.2f} "
              f"{'WIN' if won else 'loss'}")

    stands = (len(SEEDS) - wins) >= NEED_STAND   # bound-stands on >=3/4
    lifted = wins >= NEED_STAND
    if stands:
        verdict, interpretation = (
            "REPRODUCES",
            "BOUND STANDS replicates: the trace-rule redesign does not beat "
            "frozen gains on delayed_reward; NR-A-011 holds")
    elif lifted:
        verdict, interpretation = (
            "OVERTURNS",
            "BOUND LIFTED on fresh seeds: non-replication of K9; the redesign "
            "does beat frozen — a finding, reported as such")
    else:
        verdict, interpretation = (
            "INCONCLUSIVE",
            "exactly 2/4 wins: neither reproduces nor overturns; "
            "needs more seeds")
    ledger.log("verdict", {
        "t_wall": t_wall,
        "target": TARGET_ID,
        "wins": wins,
        "seeds": len(SEEDS),
        "margin": MARGIN,
        "replication_verdict": verdict,
        "interpretation": interpretation,
    })
    print(f"  -> {verdict} ({wins}/{len(SEEDS)} wins)")

    out = os.path.join(_TREE, "receipts", "repro_second_lane_k9.ndjson")
    ledger.dump(out)
    print("receipt:", out)


if __name__ == "__main__":
    main()
