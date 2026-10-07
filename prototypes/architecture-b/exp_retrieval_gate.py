"""EXP-FP-0006 — retrieval-usefulness gate (§9 item 6, B gap #2).

Preregistered spec (sealed in experiments/EXPERIMENT_REGISTRY.md before the
run; this file is the executor, not the spec).

H: gating the retrieval correction on ûhat (apply iff uhat > 0) beats
   unconditional retrieval-correction on pomaze mean episode return, and
   beats random gating at the same application rate (ranking signal test).
H0: gated <= ungated, or gated <= random-gated — ûhat has no
    decision-usable ranking signal.

Metric (EXACT): mean episode return on pomaze over 15 closed-loop episodes.
Per-seed: ΔGU = mean(G) − mean(U); ΔGR = mean(G) − mean(R).
Win rule: seed-mean ΔGU > 0 AND seed-mean ΔGR > 0 with ≥3/4 seeds agreeing
on the sign of BOTH comparisons.

Arms: U = ungated; G = RetrievalGate("uhat", threshold=0.0);
R = RetrievalGate("random", rate=r_s) with r_s the per-seed measured
application rate of arm G on the same seed, gate_seed=derive_seed(s,0,"gate").

Conditions: pomaze v1.0.0, arch_b v1, affect='none',
action_mode='active_inference', 15 episodes, identical primary seeds across
arms (paired), 4 fresh seeds {73401..73404}.

Usage:
    python3 exp_retrieval_gate.py
Receipt: flesh-pits/receipts/EXP-FP-0006.json (hash-chained).
"""

import copy
import datetime
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(FP, "flesh-pits", "experiments"))

from env_interface import (  # noqa: E402
    CONTRACT_VERSION, canonical_json, config_hash, derive_seed,
)
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from harness import write_receipt  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")
OUT_DIR = os.path.join(HERE, "experiments_out")

EXP_ID = "EXP-FP-0006"
SEEDS = [73401, 73402, 73403, 73404]
N_EPISODES = 15
ENV_NAME = "pomaze"
GATE_THRESHOLD = 0.0  # preregistered, frozen

PREREG = {
    "hypothesis": ("Gating the retrieval correction on ûhat (apply only when "
                   "uhat > 0) beats unconditional retrieval-correction on "
                   "pomaze mean episode return, and beats random gating at "
                   "the same application rate."),
    "null": ("gated <= ungated, or gated <= random-gated — ûhat has no "
             "decision-usable ranking signal."),
    "metric": ("mean episode return on pomaze, 15 closed-loop episodes; "
               "per-seed ΔGU = mean(G)−mean(U), ΔGR = mean(G)−mean(R); "
               "win rule: seed-mean ΔGU > 0 AND seed-mean ΔGR > 0 with "
               "≥3/4 seeds agreeing on the sign of BOTH comparisons."),
    "baseline": "arm U = ungated (current ArchB: correction whenever available).",
    "arms": ("U: RetrievalGate policy='ungated'; G: policy='uhat', "
             "threshold=0.0 (frozen); R: policy='random', rate=r_s = per-seed "
             "measured G application rate, gate_seed=derive_seed(s,0,'gate')."),
    "procedure": ("pomaze v1.0.0, arch_b v1, affect='none', "
                  "action_mode='active_inference', 15 episodes, identical "
                  "primary seeds across arms (paired). R runs after U and G "
                  "with r_s fixed per seed from the same seed's G run."),
    "seeds": list(SEEDS),
    "conditions": {"env": "pomaze v1.0.0", "agent": "arch_b v1",
                   "affect": "none", "action_mode": "active_inference",
                   "episodes": N_EPISODES, "gate_threshold": GATE_THRESHOLD,
                   "contract": CONTRACT_VERSION},
    "caveat": ("Closed-loop policy test: when the gate blocks a correction, "
               "measured benefit = 0 and the predictor trains on that "
               "realized outcome."),
}


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0 or dy <= 0:
        return None
    return num / math.sqrt(dx * dy)


def uhat_benefit_corr(log_path):
    """corr(uhat, benefit) from an arm-U PredictionLog JSONL (clean,
    unconditional data — the ranking check on unconditioned corrections)."""
    uhats, bens = [], []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("target") == "retrieval_usefulness":
                uhats.append(rec["prediction"])
                bens.append(rec["actual"])
    return {"n": len(uhats), "corr_uhat_benefit": corr(uhats, bens)}


def run_arm(env_name, n_eps, primary_seed, agent_seed, log_path=None,
            **gate_kwargs):
    agent = make_agent(ENVS_BY_NAME[env_name], seed=agent_seed,
                       log_path=log_path, **gate_kwargs)
    eps = run_closed_loop(env_name, agent, n_eps, primary_seed)
    return {
        "mean_return": _mean([e["return"] for e in eps]),
        "returns": [round(e["return"], 4) for e in eps],
        "gate_stats": agent.gate.stats(),
    }


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed = []
    for s in SEEDS:
        log_u = os.path.join(OUT_DIR, f"{EXP_ID}.seed{s}.U.predictions.jsonl")
        arm_u = run_arm(ENV_NAME, N_EPISODES, s, 1000 + s, log_path=log_u,
                        gate_policy="ungated")
        arm_g = run_arm(ENV_NAME, N_EPISODES, s, 1000 + s,
                        gate_policy="uhat", gate_threshold=GATE_THRESHOLD)
        r_s = arm_g["gate_stats"]["application_rate"]
        arm_r = run_arm(ENV_NAME, N_EPISODES, s, 1000 + s,
                        gate_policy="random", gate_rate=r_s,
                        gate_seed=derive_seed(s, 0, "gate"))
        dgu = arm_g["mean_return"] - arm_u["mean_return"]
        dgr = arm_g["mean_return"] - arm_r["mean_return"]
        per_seed.append({
            "seed": s,
            "U_mean_return": round(arm_u["mean_return"], 4),
            "G_mean_return": round(arm_g["mean_return"], 4),
            "R_mean_return": round(arm_r["mean_return"], 4),
            "delta_GU": round(dgu, 4),
            "delta_GR": round(dgr, 4),
            "G_gate_stats": arm_g["gate_stats"],
            "R_gate_stats": arm_r["gate_stats"],
            "U_ranking": uhat_benefit_corr(log_u),
        })
        print(f"seed {s}: U={arm_u['mean_return']:+.3f} "
              f"G={arm_g['mean_return']:+.3f} R={arm_r['mean_return']:+.3f} "
              f"dGU={dgu:+.3f} dGR={dgr:+.3f} "
              f"rate={r_s:.3f}", flush=True)

    mdgu = _mean([p["delta_GU"] for p in per_seed])
    mdgr = _mean([p["delta_GR"] for p in per_seed])
    agree_gu = sum(1 for p in per_seed if p["delta_GU"] > 0)
    agree_gr = sum(1 for p in per_seed if p["delta_GR"] > 0)
    win = mdgu > 0 and mdgr > 0 and agree_gu >= 3 and agree_gr >= 3
    if win:
        interp = (f"H SUPPORTED: seed-mean ΔGU={mdgu:+.4f} ({agree_gu}/4 seeds "
                  f"positive), seed-mean ΔGR={mdgr:+.4f} ({agree_gr}/4 "
                  f"seeds positive). ûhat gating beats unconditional "
                  f"correction AND beats chance at the matched rate — ûhat "
                  f"carries decision-usable ranking signal. Gate INTEGRATED.")
    elif mdgr <= 0:
        interp = (f"NEGATIVE: seed-mean ΔGR={mdgr:+.4f} (G ≤ R on "
                  f"{4 - agree_gr}/4 seeds) — ûhat gating does not beat "
                  f"random gating at the same rate. The predictor has no "
                  f"ranking signal worth gating on; it stays EXECUTED. "
                  f"Gap #2 closed by rejection.")
    else:
        interp = (f"MIXED: seed-mean ΔGU={mdgu:+.4f} ({agree_gu}/4), "
                  f"seed-mean ΔGR={mdgr:+.4f} ({agree_gr}/4). ûhat ranks "
                  f"but gating does not beat unconditional correction — "
                  f"rate/threshold effect; investigate.")

    result = {
        "experiment_id": EXP_ID,
        "config": PREREG["conditions"],
        "config_hash": config_hash(PREREG["conditions"]),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [],
        "summary": {
            "per_seed": per_seed,
            "seed_mean_delta_GU": round(mdgu, 4),
            "seed_mean_delta_GR": round(mdgr, 4),
            "seeds_agree_GU": agree_gu,
            "seeds_agree_GR": agree_gr,
            "verdict": "SUPPORTED" if win else (
                "NEGATIVE (no ranking signal)" if mdgr <= 0 else "MIXED"),
        },
    }
    path = write_receipt(
        result, RECEIPTS_DIR,
        hypothesis=PREREG["hypothesis"], null=PREREG["null"],
        preregistered_metric=PREREG["metric"],
        baseline=PREREG["baseline"],
        conditions=json.dumps(PREREG["conditions"], sort_keys=True),
        interpretation=interp,
        limitations=("pomaze only; 4 seeds; affect='none' isolates the "
                     "world model; closed-loop policy test — blocked "
                     "corrections train the predictor on benefit=0 "
                     "(realized outcome); random arm rate r_s is the "
                     "per-seed measured G application rate (conditional "
                     "on correction availability)."),
    )
    print(f"\nverdict: {result['summary']['verdict']}\n  {path}")
    print(f"seed-mean ΔGU={mdgu:+.4f} ΔGR={mdgr:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
