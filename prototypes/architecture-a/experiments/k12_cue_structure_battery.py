"""K12 — CUE-STRUCTURE BATTERY FOR K10 (NR-A-007 bounds).

Preregistration: prototypes/architecture-a/receipts/preregistration_K12-*.json
(sealed 2026-10-07T09:45:00Z, BEFORE this driver was written).

Battery axes (all K10 protocol: 12 eps x 40 steps, a0/a1, neutral
specialists, capacity=2, gain_lr=0.15, theta=0.45, original delta rule,
CueIndexedArbitrator; frozen = same class frozen=True):
  A: cue cardinality (n=2 anchor / n=4 / n=8)
  B: contingency type (magnitude-fixed / distractor-fixed / mag x interaction)
  C: cue observability timing (episodic / delayed / t0only)
  D: cue reliability (noise 0.25 / 0.50)
Env: experiments/k12_cue_battery_envs.py (K12CueBattery; canonical knobs
byte-equivalent to changing_rule v1.0.0 — verified in preflight).

Per condition: main probe (context_fn = obs.get("cue")) + control probe
(context_fn = None const) x learned/frozen x 4 fresh seeds.
Gate (K10 confound rule): CUE-CAUSED LIFT iff main_wins >= 3/4 AND
control_wins <= 1/4; ADAPTER-CAUSED iff main >= 3/4 AND control >= 3/4;
BOUND STANDS iff main <= 1/4; 2/4 = INCONCLUSIVE.

Receipts are written with experiments/harness.py write_receipt (fail
closed on existing paths). Does NOT import the original K10 script.
"""
import datetime
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TREE = os.path.dirname(_HERE)
_FLESH = os.path.dirname(os.path.dirname(_TREE))
# Phase-1 path order (K10 convention): _TREE/envs.py shadows the experiments
# envs package, which k5_multitask_generalization requires.
sys.path.insert(0, os.path.join(_FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(_FLESH, "experiments"))
sys.path.insert(0, _TREE)
sys.path.insert(0, _HERE)

from tick import WorkspaceTick
from attention_cue import CueIndexedArbitrator
from k5_multitask_generalization import CanonicalAdapter, neutral_specialists
from k12_cue_battery_envs import K12CueBattery
# NOTE: harness is imported lazily in phase 2 (write_receipts) with the
# `envs` shadow re-pointed at experiments/envs — see below.

MARGIN = 1.30
NEED = 3
EPISODES = 12
STEPS = 40
CHANNELS = ["a0", "a1"]
CHANNEL_TO_ACTION = {"a0": 0, "a1": 1}

CONDITIONS = {
    "K12-A1": dict(env_kwargs={}, seeds=(41001, 41002, 41003, 41004)),
    "K12-A2": dict(env_kwargs={"n_cues": 4}, seeds=(42001, 42002, 42003, 42004)),
    "K12-A3": dict(env_kwargs={"n_cues": 8}, seeds=(43001, 43002, 43003, 43004)),
    "K12-B1": dict(env_kwargs={"contingency": "magnitude", "flips": False},
                   seeds=(51001, 51002, 51003, 51004)),
    "K12-B2": dict(env_kwargs={"contingency": "distractor", "flips": False},
                   seeds=(52001, 52002, 52003, 52004)),
    "K12-B3": dict(env_kwargs={"contingency": "mag_interaction"},
                   seeds=(53001, 53002, 53003, 53004)),
    "K12-C1": dict(env_kwargs={"timing": "episodic"},
                   seeds=(61001, 61002, 61003, 61004)),
    "K12-C2": dict(env_kwargs={"timing": "delayed"},
                   seeds=(62001, 62002, 62003, 62004)),
    "K12-C3": dict(env_kwargs={"timing": "t0only"},
                   seeds=(63001, 63002, 63003, 63004)),
    "K12-D2": dict(env_kwargs={"cue_noise": 0.25},
                   seeds=(72001, 72002, 72003, 72004)),
    "K12-D3": dict(env_kwargs={"cue_noise": 0.50},
                   seeds=(73001, 73002, 73003, 73004)),
}


def run_arm(seed, frozen, with_cue, env_kwargs):
    """One K10 arm (learned/frozen x cue-in/cue-out) on one seed."""
    context_fn = (lambda obs: obs.get("cue")) if with_cue else (
        lambda obs: None)
    wk = WorkspaceTick(
        list(CHANNELS), neutral_specialists(list(CHANNELS), seed),
        capacity=2, frozen_gains=frozen, gain_lr=0.15,
        ignition_kwargs={"theta": 0.45},
        arbitrator_cls=CueIndexedArbitrator, context_fn=context_fn)
    assert isinstance(wk.arbitrator, CueIndexedArbitrator)
    env = CanonicalAdapter(K12CueBattery(**env_kwargs), seed,
                           CHANNEL_TO_ACTION, n_episodes=EPISODES)
    for _ in range(EPISODES * STEPS):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), wk.arbitrator.all_context_gains(),
            len(wk.arbitrator.gain_history))


def classify(main_wins, ctrl_wins):
    if main_wins >= NEED and ctrl_wins <= (4 - NEED):
        return "CUE-CAUSED LIFT"
    if main_wins >= NEED and ctrl_wins >= NEED:
        return "ADAPTER-CAUSED (not cue-caused)"
    if main_wins <= 1:
        return "BOUND STANDS"
    return "INCONCLUSIVE"


def verdict_for(eid, main_rs, ctrl_rs):
    main_wins = sum(1 for r in main_rs if r >= MARGIN)
    ctrl_wins = sum(1 for r in ctrl_rs if r >= MARGIN)
    cls = classify(main_wins, ctrl_wins)
    interp = {
        "CUE-CAUSED LIFT": "main gaps with cue input and clean control: the lift is cue-caused",
        "ADAPTER-CAUSED (not cue-caused)": "main AND control both gap: the lift comes from the learning loop, not the cue",
        "BOUND STANDS": "main shows no gap: the NR-A-007 bound reasserts on this cue structure",
        "INCONCLUSIVE": "exactly 2/4 main wins: needs more seeds",
    }[cls]
    return main_wins, ctrl_wins, cls, interp


def load_write_receipt():
    """Phase-2 import: re-point the `envs` shadow at experiments/envs so
    harness.py's `from envs import ALL_ENVS` resolves to the package, not
    prototypes/architecture-a/envs.py. Phase-1 modules are unaffected
    (already imported)."""
    for mod in [m for m in list(sys.modules)
                if m == "envs" or m.startswith("envs.")]:
        del sys.modules[mod]
    for p in (_TREE, _HERE):
        while p in sys.path:
            sys.path.remove(p)
    if os.path.join(_FLESH, "experiments") not in sys.path:
        sys.path.insert(0, os.path.join(_FLESH, "experiments", "envs"))
        sys.path.insert(0, os.path.join(_FLESH, "experiments"))
    from harness import write_receipt
    return write_receipt


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipts_dir = os.path.join(_TREE, "receipts")
    results = {}
    for eid, spec in CONDITIONS.items():
        kw = spec["env_kwargs"]
        main_rs, ctrl_rs = [], []
        seed_rows = []
        for seed in spec["seeds"]:
            tl, gains_l, n_upd = run_arm(seed, False, True, kw)
            tf, _, _ = run_arm(seed, True, True, kw)
            r = tl / tf if tf > 0 else float("inf")
            main_rs.append(r)
            tl2, _, _ = run_arm(seed, False, False, kw)
            tf2, _, _ = run_arm(seed, True, False, kw)
            r2 = tl2 / tf2 if tf2 > 0 else float("inf")
            ctrl_rs.append(r2)
            seed_rows.append({
                "seed": seed,
                "main_learned": round(tl, 2), "main_frozen": round(tf, 2),
                "main_R": round(r, 3), "main_win": bool(r >= MARGIN),
                "ctrl_learned": round(tl2, 2), "ctrl_frozen": round(tf2, 2),
                "ctrl_R": round(r2, 3), "ctrl_gap": bool(r2 >= MARGIN),
            })
            print(f"  {eid} seed={seed}: main R={r:.2f} "
                  f"{'WIN' if r >= MARGIN else '---'}  ctrl R={r2:.2f} "
                  f"{'GAP' if r2 >= MARGIN else '---'}", flush=True)
        mw, cw, cls, interp = verdict_for(eid, main_rs, ctrl_rs)
        print(f"  {eid}: main {mw}/4 ctrl-gaps {cw}/4 -> {cls}")
        results[eid] = {
            "env_kwargs": kw, "seeds": list(spec["seeds"]),
            "seed_rows": seed_rows, "main_wins": mw, "control_wins": cw,
            "verdict": cls, "interpretation": interp,
        }

    # ---- receipts (fail closed; never overwrite) ----
    write_receipt = load_write_receipt()
    config = {"episodes": EPISODES, "steps_per_episode": STEPS,
              "channels": CHANNELS, "capacity": 2, "gain_lr": 0.15,
              "theta": 0.45, "arbitrator": "CueIndexedArbitrator",
              "update_rule": "original delta rule, baseline 0.5",
              "env": "k12_cue_battery v1.0.0",
              "prereg": "preregistration_<ID>.json sealed 2026-10-07T09:45Z"}
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()
    for eid, res in results.items():
        with open(os.path.join(receipts_dir,
                               f"preregistration_{eid}.json")) as f:
            prereg = json.load(f)
        summary = {
            "env_kwargs": res["env_kwargs"],
            "seeds": res["seeds"],
            "seed_rows": res["seed_rows"],
            "main_wins": res["main_wins"], "control_wins": res["control_wins"],
            "margin": MARGIN, "need": NEED,
            "verdict": res["verdict"],
        }
        path = write_receipt(
            {"experiment_id": eid, "config": config,
             "config_hash": config_hash, "primary_seed": res["seeds"][0],
             "started_utc": t0, "episodes": EPISODES, "summary": summary},
            receipts_dir,
            hypothesis=prereg["hypothesis"],
            null=prereg["null_hypothesis"],
            preregistered_metric=prereg["preregistered_metric"],
            baseline=prereg["baseline"],
            conditions=prereg["env"] + "; " + prereg["protocol"],
            interpretation=(f"{res['verdict']}: {res['interpretation']} "
                            f"(prediction was: {prereg['prediction']})"),
            limitations=("12 episodes x 40 steps; neutral specialists "
                         "(bandit isolation, no obs input to stimuli); 4 "
                         "seeds/condition; flips every 5 episodes where "
                         "enabled; 10% reward noise."),
        )
        print("receipt:", path)


if __name__ == "__main__":
    main()
