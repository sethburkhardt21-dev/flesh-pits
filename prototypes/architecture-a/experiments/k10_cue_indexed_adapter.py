"""K10 — CUE-INDEXED GAIN ADAPTER (NR-A-007 candidate).

PREREGISTRATION (written 2026-10-07T03:56 EDT, BEFORE implementation/execution;
canonical copy: receipts/prereg_k10_cue_indexed_adapter.json):

  Hypothesis (H1): context-conditioned gains with the cue in the input let
    learned gains beat frozen gains on cue-conditioned changing_rule
    (R >= 1.30). Mechanism: per-cue delta-rule learning acquires each
    cue's correct action; after each rule flip the per-cue vectors
    re-learn — the K4/P3 reversal-tracking mechanism, now cue-indexed.
  Null (H0): no gap even with cue input (R < 1.30 on >= 2 of 4 seeds) ->
    NR-A-007's bound is deeper than input conditioning.
  Metric: R = total(learned) / total(frozen) per seed, 12 episodes x 40
    steps on canonical changing_rule v1.0.0.
  Gate: BOUND LIFTED iff (i) main probe R >= 1.30 on >= 3 of 4 fresh
    seeds {5555, 6666, 7777, 8888} AND (ii) control probe (cue withheld,
    constant context) shows NO gap (R < 1.30 on >= 3/4). If (i) holds
    but (ii) fails -> CONFOUND (module bug), do NOT claim. If (i)
    fails -> BOUND STANDS -> NR-A-012.
  Config: K5 P4 protocol in every hyperparameter (channels a0/a1, neutral
    specialists, capacity=2, gain_lr=0.15, theta=0.45); the update rule is
    the ORIGINAL delta rule (baseline 0.5); the ONLY change is
    architectural (cue-indexed gains via CueIndexedArbitrator +
    context_fn). Frozen = same class, frozen=True.

Probes:
  B1_cue_indexed: context_fn = lambda obs: int(obs["cue"])  (cue in input)
  B2_cue_withheld: context_fn = lambda obs: None            (control)

Kill (per probe): R < 1.30 on majority of seeds.
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
from attention_cue import CueIndexedArbitrator
from changing_rule import ChangingRule
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)

SEEDS = (5555, 6666, 7777, 8888)
THETA = 0.45
MARGIN = 1.30
NEED = 3
GAIN_LR = 0.15
EXPERIMENT_ID = "K10-cue-indexed-adapter"
N_EPISODES = 12


def probe_cue_indexed(seed, frozen, with_cue):
    """K5 P4 protocol, but the arbitrator is cue-indexed; the cue reaches
    the gains iff with_cue."""
    channels = ["a0", "a1"]
    context_fn = (lambda obs: int(obs["cue"])) if with_cue else (
        lambda obs: None)
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=2, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=CueIndexedArbitrator,
                       context_fn=context_fn)
    assert isinstance(wk.arbitrator, CueIndexedArbitrator)
    env = CanonicalAdapter(ChangingRule(), seed, {"a0": 0, "a1": 1},
                           n_episodes=N_EPISODES)
    for _ in range(N_EPISODES * ChangingRule.STEPS):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), wk.arbitrator.all_context_gains(),
            len(wk.arbitrator.gain_history))


def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def run_probe(pname, with_cue, emit, printer):
    printer(f"--- {pname} (cue_in_input={with_cue})")
    wins = 0
    for seed in SEEDS:
        tl, ctx_l, n_upd = probe_cue_indexed(seed, False, with_cue)
        tf, _ctx_f, _ = probe_cue_indexed(seed, True, with_cue)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        emit("seed_result", {
            "probe": pname, "seed": seed, "cue_in_input": bool(with_cue),
            "learned_total": round(tl, 2), "frozen_total": round(tf, 2),
            "ratio": round(r, 3), "win": bool(win),
            "learned_context_gains": {
                str(k): {c: round(v, 3) for c, v in vec.items()}
                for k, vec in ctx_l.items()},
            "n_gain_updates": n_upd,
        })
        printer(f"  seed={seed}: learned={tl:.1f} frozen={tf:.1f} "
                f"R={r:.2f} {'WIN' if win else 'loss'}")
    return wins


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
        "hypothesis": "H1: cue-indexed gains beat frozen gains on "
                      "cue-conditioned changing_rule (R >= 1.30, >=3/4)",
        "null": "H0: no gap even with cue -> NR-A-007 stands",
        "metric": "R = total(learned)/total(frozen) per seed",
        "gate": f"BOUND LIFTED iff main R >= {MARGIN} on >= {NEED}/"
                f"{len(SEEDS)} AND control R < {MARGIN} on >= {NEED}/"
                f"{len(SEEDS)}; confound rule stated",
        "config": {"env": "changing_rule v1.0.0",
                   "episodes": N_EPISODES, "steps": ChangingRule.STEPS,
                   "channels": ["a0", "a1"],
                   "specialists": "neutral (K5 P4 identical)",
                   "capacity": 2, "gain_lr": GAIN_LR, "theta": THETA,
                   "update_rule": "original delta rule, baseline 0.5",
                   "arbitrator": "CueIndexedArbitrator",
                   "frozen": "same class, frozen=True, gains pinned 1.0"},
        "prereg_file": "receipts/prereg_k10_cue_indexed_adapter.json",
    })

    print(f"{EXPERIMENT_ID}")
    wins_main = run_probe("B1_cue_indexed", True, emit, print)
    wins_ctrl = run_probe("B2_cue_withheld", False, emit, print)

    main_pass = wins_main >= NEED
    ctrl_clean = (len(SEEDS) - wins_ctrl) >= NEED  # no gap on >=3/4
    if main_pass and ctrl_clean:
        verdict, interp = True, ("BOUND LIFTED: cue-indexed gains beat "
                                 "frozen with cue input; control shows no "
                                 "gap without it")
    elif main_pass and not ctrl_clean:
        verdict, interp = False, ("CONFOUND: control also shows a gap -> "
                                  "module bug, do NOT claim; investigate")
    else:
        verdict, interp = False, ("BOUND STANDS (NR-A-007 architectural): "
                                  "cue input does not create a gap; "
                                  "record NR-A-012")
    emit("verdict", {"main_wins": wins_main, "control_wins": wins_ctrl,
                     "need": NEED, "margin": MARGIN,
                     "main_pass": bool(main_pass),
                     "control_clean": bool(ctrl_clean),
                     "bound_lifted": bool(verdict),
                     "interpretation": interp})
    print(f"  main: {wins_main}/{len(SEEDS)}  control gaps: "
          f"{wins_ctrl}/{len(SEEDS)}  -> "
          f"{'BOUND LIFTED' if verdict else 'BOUND STANDS / CONFOUND'}")

    out = os.path.join(TREE, "receipts", "k10_cue_indexed_adapter.ndjson")
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print("receipt:", out)


if __name__ == "__main__":
    main()
