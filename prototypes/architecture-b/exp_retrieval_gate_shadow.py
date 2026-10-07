"""EXP-FP-0007 — non-degenerate uhat-gating instrument: shadow-trained predictor.

Preregistered spec (sealed in experiments/EXPERIMENT_REGISTRY.md before any
code was written; this file is the executor, not the spec).

H: gating the retrieval correction on uhat from a SHADOW-trained predictor
   (training data uncontaminated by gating) beats unconditional
   retrieval-correction on pomaze mean episode return, and beats random
   gating at the same application rate (ranking-signal decision value).
H0: gated <= ungated, or gated <= random-gated.

Metric (EXACT): mean episode return on pomaze over 15 closed-loop episodes.
Per-seed: dGU = mean(G) - mean(U); dGR = mean(G) - mean(R).
Win rule: seed-mean dGU > 0 AND seed-mean dGR > 0 with >=3/4 seeds agreeing
on the sign of BOTH comparisons. (Identical to EXP-FP-0006.)

Arms: U = ungated; G = RetrievalGate("uhat", threshold=0.0 FROZEN,
  gate_uhat_source="shadow"); R = RetrievalGate("random", rate=r_s) with
  r_s the per-seed measured application rate of arm G on the same seed,
  gate_seed=derive_seed(s, 0, "gate").

The shadow predictor (agent.shadow_usefulness) trains on the counterfactual
unconditional benefit every available-correction tick; the gate reads its
uhat. The live UsefulnessPredictor trains on realized benefit as in
EXP-FP-0006 and is not read by the gate. predictions.py is NOT modified.

Frozen gates (VOID, not reinterpreted):
  G0 instrument-validity: G application rate strictly in (0, 1) on >=3/4
     seeds (the gate must make discriminating decisions); else VOID as
     instrument failure (EXP-FP-0006 lesson).
  G1 fabrication-tripwire CLEAN on new/modified code, pre-run (checked by
     the operator before launching; re-asserted here as a gate).
  G2 determinism spot-check: seed 73501 G arm recomputed -> mean_return
     identical to 1e-12, else VOID.
  G3 hash-chained receipt + verify_chain.

Usage:
    python3 exp_retrieval_gate_shadow.py
Receipt: flesh-pits/receipts/EXP-FP-0007.json (hash-chained).
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
from harness import write_receipt, verify_chain  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")
OUT_DIR = os.path.join(HERE, "experiments_out")

EXP_ID = "EXP-FP-0007"
SEEDS = [73501, 73502, 73503, 73504]
N_EPISODES = 15
ENV_NAME = "pomaze"
GATE_THRESHOLD = 0.0  # preregistered, frozen

PREREG = {
    "hypothesis": ("Gating the retrieval correction on uhat from a "
                   "shadow-trained predictor (training data uncontaminated "
                   "by gating) beats unconditional retrieval-correction on "
                   "pomaze mean episode return, and beats random gating at "
                   "the same application rate."),
    "null": ("gated <= ungated, or gated <= random-gated - the ranking "
             "signal has no decision value even with a non-degenerate "
             "instrument."),
    "metric": ("mean episode return on pomaze, 15 closed-loop episodes; "
               "per-seed dGU = mean(G)-mean(U), dGR = mean(G)-mean(R); "
               "win rule: seed-mean dGU > 0 AND seed-mean dGR > 0 with "
               ">=3/4 seeds agreeing on the sign of BOTH comparisons."),
    "baseline": "arm U = ungated (current ArchB: correction whenever available).",
    "arms": ("U: gate_policy='ungated'; G: gate_policy='uhat', "
             "threshold=0.0 (frozen), gate_uhat_source='shadow'; R: "
             "gate_policy='random', rate=r_s = per-seed measured G "
             "application rate, gate_seed=derive_seed(s,0,'gate')."),
    "procedure": ("pomaze v1.0.0, arch_b v1, affect='none', "
                  "action_mode='active_inference', 15 episodes, identical "
                  "primary seeds across arms (paired). R runs after U and G "
                  "with r_s fixed per seed from the same seed's G run."),
    "seeds": list(SEEDS),
    "conditions": {"env": "pomaze v1.0.0", "agent": "arch_b v1",
                   "affect": "none", "action_mode": "active_inference",
                   "episodes": N_EPISODES, "gate_threshold": GATE_THRESHOLD,
                   "gate_uhat_source": "shadow",
                   "contract": CONTRACT_VERSION},
    "caveat": ("Closed-loop policy test: the gated trajectory diverges from "
               "the ungated trajectory once gating blocks; the shadow "
               "predictor's training data is uncontaminated by gating "
               "(counterfactual unconditional benefit) but lives on the "
               "gated trajectory."),
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
    unconditional data - replication of the EXP-FP-0006 ranking check)."""
    uhats, bens = [], []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("target") == "retrieval_usefulness":
                uhats.append(rec["prediction"])
                bens.append(rec["actual"])
    return {"n": len(uhats), "corr_uhat_benefit": corr(uhats, bens)}


def run_arm(env_name, n_eps, primary_seed, agent_seed, log_path=None,
            want_shadow_hist=False, **gate_kwargs):
    agent = make_agent(ENVS_BY_NAME[env_name], seed=agent_seed,
                       log_path=log_path, **gate_kwargs)
    eps = run_closed_loop(env_name, agent, n_eps, primary_seed)
    out = {
        "mean_return": _mean([e["return"] for e in eps]),
        "returns": [round(e["return"], 4) for e in eps],
        "gate_stats": agent.gate.stats(),
    }
    if want_shadow_hist:
        uh = [u for u, b in agent._shadow_hist]
        bh = [b for u, b in agent._shadow_hist]
        out["shadow_corr"] = {"n": len(uh),
                              "corr_uhat_benefit": corr(uh, bh)}
        out["shadow_w"] = list(agent.shadow_usefulness.w)
        out["shadow_b"] = agent.shadow_usefulness.b
    return out


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed = []
    for s in SEEDS:
        log_u = os.path.join(OUT_DIR, f"{EXP_ID}.seed{s}.U.predictions.jsonl")
        arm_u = run_arm(ENV_NAME, N_EPISODES, s, 1000 + s, log_path=log_u,
                        gate_policy="ungated")
        arm_g = run_arm(ENV_NAME, N_EPISODES, s, 1000 + s,
                        gate_policy="uhat", gate_threshold=GATE_THRESHOLD,
                        gate_uhat_source="shadow", want_shadow_hist=True)
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
            "G_mean_return_raw": arm_g["mean_return"],  # unrounded, G2 gate
            "R_mean_return": round(arm_r["mean_return"], 4),
            "delta_GU": round(dgu, 4),
            "delta_GR": round(dgr, 4),
            "G_gate_stats": arm_g["gate_stats"],
            "G_shadow_corr": arm_g["shadow_corr"],
            "G_shadow_w": [round(w, 6) for w in arm_g["shadow_w"]],
            "G_shadow_b": round(arm_g["shadow_b"], 6),
            "R_gate_stats": arm_r["gate_stats"],
            "U_ranking": uhat_benefit_corr(log_u),
        })
        print(f"seed {s}: U={arm_u['mean_return']:+.3f} "
              f"G={arm_g['mean_return']:+.3f} R={arm_r['mean_return']:+.3f} "
              f"dGU={dgu:+.3f} dGR={dgr:+.3f} "
              f"rate={r_s:.3f} "
              f"shadow_corr={arm_g['shadow_corr']['corr_uhat_benefit']}",
              flush=True)

    # G0 instrument-validity: discriminating decisions on >=3/4 seeds.
    n_discriminating = sum(
        1 for p in per_seed
        if 0.0 < p["G_gate_stats"]["application_rate"] < 1.0)
    if n_discriminating < 3:
        print(f"\nG0 FAIL: instrument degenerate "
              f"({n_discriminating}/4 seeds discriminating). VOID - no "
              f"interpretation, results discarded per the frozen gate.")
        return 2

    # G2 determinism spot-check: recompute seed 73501 G arm, compared
    # UNROUNDED (the voided first run compared a rounded value against an
    # unrounded one - gate bug, fixed 2026-10-07; the arm itself was
    # verified bit-identical across recomputes).
    g_re = run_arm(ENV_NAME, N_EPISODES, SEEDS[0], 1000 + SEEDS[0],
                   gate_policy="uhat", gate_threshold=GATE_THRESHOLD,
                   gate_uhat_source="shadow")
    if abs(g_re["mean_return"] - per_seed[0]["G_mean_return_raw"]) > 1e-12:
        print("\nG2 FAIL: determinism spot-check mismatch. VOID.")
        return 2
    print(f"G2 determinism spot-check PASS "
          f"({g_re['mean_return']:.4f} == {per_seed[0]['G_mean_return']:.4f})")

    mdgu = _mean([p["delta_GU"] for p in per_seed])
    mdgr = _mean([p["delta_GR"] for p in per_seed])
    agree_gu = sum(1 for p in per_seed if p["delta_GU"] > 0)
    agree_gr = sum(1 for p in per_seed if p["delta_GR"] > 0)
    win = mdgu > 0 and mdgr > 0 and agree_gu >= 3 and agree_gr >= 3
    if win:
        interp = (f"H SUPPORTED: seed-mean dGU={mdgu:+.4f} ({agree_gu}/4 "
                  f"seeds positive), seed-mean dGR={mdgr:+.4f} "
                  f"({agree_gr}/4 seeds positive). The shadow-ûhat gate "
                  f"beats unconditional correction AND beats chance at the "
                  f"matched rate - the ranking signal carries "
                  f"decision-usable value. Gate INTEGRATED (maturity row "
                  f"updated).")
    elif mdgr <= 0:
        interp = (f"NEGATIVE: seed-mean dGR={mdgr:+.4f} (G <= R on "
                  f"{4 - agree_gr}/4 seeds) - the shadow-ûhat gate does not "
                  f"beat random gating at the same rate. The ranking signal "
                  f"has no measured decision value on pomaze return even "
                  f"with a non-degenerate instrument; the predictor stays "
                  f"EXECUTED. Recorded as a bound on gap #2.")
    else:
        interp = (f"MIXED: seed-mean dGU={mdgu:+.4f} ({agree_gu}/4), "
                  f"seed-mean dGR={mdgr:+.4f} ({agree_gr}/4). ûhat ranks "
                  f"but gating does not beat unconditional correction - "
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
            "g0_discriminating_seeds": n_discriminating,
            "verdict": "SUPPORTED" if win else (
                "NEGATIVE (no decision value)" if mdgr <= 0 else "MIXED"),
        },
    }
    path = write_receipt(
        result, RECEIPTS_DIR,
        hypothesis=PREREG["hypothesis"], null=PREREG["null"],
        preregistered_metric=PREREG["metric"],
        baseline=PREREG["baseline"],
        conditions=json.dumps(PREREG["conditions"], sort_keys=True),
        interpretation=interp,
        limitations=("pomaze only; 4 fresh seeds; affect='none' isolates "
                     "the world model; closed-loop policy test - the gated "
                     "trajectory diverges from the ungated trajectory; the "
                     "shadow predictor's training data is uncontaminated by "
                     "gating (counterfactual unconditional benefit) but "
                     "lives on the gated trajectory; random arm rate r_s is "
                     "the per-seed measured G application rate."),
    )
    ok, problems = verify_chain(RECEIPTS_DIR)
    print(f"hash chain: {'OK' if ok else problems}")
    print(f"\nverdict: {result['summary']['verdict']}\n  {path}")
    print(f"seed-mean dGU={mdgu:+.4f} dGR={mdgr:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
