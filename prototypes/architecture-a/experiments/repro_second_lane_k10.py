"""REPRO-SECOND-LANE-K10 — independent replication of K10.

Spec: PREREGISTRATION receipts/prereg_repro_second_lane_k10.json
(sealed 2026-10-07T07:58:00Z, BEFORE this driver was written).

Protocol is the K10 spec from the prereg: canonical changing_rule
v1.0.0, 12 episodes x 40 steps, CueIndexedArbitrator (original
constant-baseline delta rule unchanged), neutral specialists,
capacity=2, gain_lr=0.15, theta=0.45. Two probes:
  B1 cue_indexed   context_fn = obs -> int(obs["cue"])   (cue in input)
  B2 cue_withheld  context_fn = obs -> None              (control)
Metric per probe: R = total(learned)/total(frozen) per seed;
WIN/GAP iff R >= 1.30. Fresh seeds {91615, 91616, 91617, 91618}
(disjoint from original {5555, 6666, 7777, 8888}).

Replication verdict: B1 REPRODUCES iff wins >= 3/4 (BOUND LIFTED);
B1 OVERTURNS iff wins <= 1/4. B2 replicates iff gaps <= 1/4;
CONFOUND iff gaps >= 3/4. Exactly 2/4 on either probe = INCONCLUSIVE.

Imports only the architecture modules under test. Does NOT import the
original k10_cue_indexed_adapter.py script.
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
from attention_cue import CueIndexedArbitrator
from changing_rule import ChangingRule
from k5_multitask_generalization import CanonicalAdapter, neutral_specialists

EXPERIMENT_ID = "REPRO-SECOND-LANE-K10"
TARGET_ID = "K10-cue-indexed-adapter"
SEEDS = (91615, 91616, 91617, 91618)
CHANNELS = ("a0", "a1")
EPISODES = 12
MARGIN = 1.30
NEED = 3   # 3/4 to reproduce the probe's verdict; <=1/4 overturns


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


def replicate_condition(seed, frozen, with_cue):
    """One K10 condition (learned/frozen x cue-in/cue-out) on one seed."""
    context_fn = (lambda obs: int(obs["cue"])) if with_cue else (
        lambda obs: None)
    wk = WorkspaceTick(
        list(CHANNELS), neutral_specialists(list(CHANNELS), seed),
        capacity=2, frozen_gains=frozen, gain_lr=0.15,
        ignition_kwargs={"theta": 0.45},
        arbitrator_cls=CueIndexedArbitrator, context_fn=context_fn)
    assert isinstance(wk.arbitrator, CueIndexedArbitrator), (
        "protocol requires the cue-indexed arbitrator")
    env = CanonicalAdapter(ChangingRule(), seed, {"a0": 0, "a1": 1},
                           n_episodes=EPISODES)
    total_ticks = EPISODES * ChangingRule.STEPS
    for _ in range(total_ticks):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), wk.arbitrator.all_context_gains(),
            len(wk.arbitrator.gain_history))


def run_probe(probe_name, with_cue, ledger, t_wall):
    hits = 0
    for seed in SEEDS:
        learned_total, context_gains, n_updates = replicate_condition(
            seed, frozen=False, with_cue=with_cue)
        frozen_total, _, _ = replicate_condition(
            seed, frozen=True, with_cue=with_cue)
        ratio = (learned_total / frozen_total
                 if frozen_total > 0 else float("inf"))
        hit = ratio >= MARGIN
        hits += int(hit)
        ledger.log("seed_result", {
            "t_wall": t_wall,
            "target": TARGET_ID,
            "probe": probe_name,
            "seed": seed,
            "cue_in_input": bool(with_cue),
            "learned_total": round(learned_total, 2),
            "frozen_total": round(frozen_total, 2),
            "ratio": round(ratio, 3),
            "hit": bool(hit),
            "learned_context_gains": {
                str(ctx): {c: round(g, 3) for c, g in vec.items()}
                for ctx, vec in context_gains.items()},
            "n_gain_updates": n_updates,
        })
        print(f"  {probe_name} seed={seed} learned={learned_total:.1f} "
              f"frozen={frozen_total:.1f} R={ratio:.2f} "
              f"{'HIT' if hit else 'miss'}")
    return hits


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    t_wall = t0.isoformat()
    ledger = ChainWriter(EXPERIMENT_ID)

    ledger.log("preregistration", {
        "t_wall": t_wall,
        "target": TARGET_ID,
        "prereg_file": "receipts/prereg_repro_second_lane_k10.json",
        "seal_sha256": "48453d1a5cb851003166bf1ebca056f3e28d7340966ab44fbb7751d6cc0c2c00",
        "note": ("independent second-lane driver; preregistration sealed "
                 "before this driver was written"),
        "hypothesis": "both K10 verdicts replicate on fresh seeds",
        "gate": (f"B1 WIN iff R >= {MARGIN} (>= {NEED}/{len(SEEDS)} to "
                 f"reproduce); B2 no-gap iff gaps <= {len(SEEDS) - NEED}/"
                 f"{len(SEEDS)}"),
        "seeds": list(SEEDS),
    })

    main_hits = run_probe("B1_cue_indexed", True, ledger, t_wall)
    ctrl_hits = run_probe("B2_cue_withheld", False, ledger, t_wall)

    b1_reproduces = main_hits >= NEED
    b1_overturns = main_hits <= len(SEEDS) - NEED
    b2_clean = ctrl_hits <= len(SEEDS) - NEED      # no gap on >=3/4
    b2_confound = ctrl_hits >= NEED

    if b1_reproduces and b2_clean:
        verdict, interpretation = (
            "REPRODUCES",
            "both probes replicate: cue-indexed gains beat frozen with cue "
            "input; control shows no gap without it (NR-A-007 conditionally "
            "lifted, holds)")
    elif b1_reproduces and b2_confound:
        verdict, interpretation = (
            "CONFOUND",
            "B1 replicates but the control also shows a gap: module bug "
            "suspected; do NOT claim; investigate")
    elif b1_overturns:
        verdict, interpretation = (
            "OVERTURNS",
            "B1 fails on fresh seeds (<=1/4 wins): K10's lift does not "
            "replicate — a finding, reported as such")
    else:
        verdict, interpretation = (
            "INCONCLUSIVE",
            "exactly 2/4 on a probe: neither reproduces nor overturns; "
            "needs more seeds")

    ledger.log("verdict", {
        "t_wall": t_wall,
        "target": TARGET_ID,
        "main_hits": main_hits,
        "control_hits": ctrl_hits,
        "need": NEED,
        "margin": MARGIN,
        "b1_reproduces": bool(b1_reproduces),
        "b2_clean": bool(b2_clean),
        "replication_verdict": verdict,
        "interpretation": interpretation,
    })
    print(f"  B1 hits: {main_hits}/{len(SEEDS)}  B2 gaps: "
          f"{ctrl_hits}/{len(SEEDS)}  -> {verdict}")

    out = os.path.join(_TREE, "receipts", "repro_second_lane_k10.ndjson")
    ledger.dump(out)
    print("receipt:", out)


if __name__ == "__main__":
    main()
