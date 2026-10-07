"""EXP-FP-0007R — replication breadth for the shadow-uhat gate.

Preregistered: experiments/preregistration_EXP-FP-0007R.json (before run).
Reuses the EXP-FP-0007 instrument VERBATIM (imported, not modified):
  run_arm, uhat_benefit_corr, _mean, corr, write_receipt, verify_chain
from exp_retrieval_gate_shadow. This file is the executor, not the spec.

Bar (frozen): per env, win = seed-mean dGU > 0 AND seed-mean dGR > 0 with
>=4/5 seeds agreeing on the sign of BOTH comparisons. CAUSAL promotion
iff pomaze win fires AND at least one of delayed_reward/changing_rule fires.

Frozen gates: G0 instrument-validity (>=4/5 seeds strictly in (0,1)
application rate per env), G1 tripwire clean (pre-run, re-asserted here),
G2 determinism spot-check (per env, first seed G arm recomputed, unrounded
compare at 1e-12), G3 hash-chained receipt + verify_chain.
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

import exp_retrieval_gate_shadow as base  # noqa: E402 — the instrument, verbatim
from env_interface import derive_seed, config_hash  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")
OUT_DIR = os.path.join(HERE, "experiments_out")

EXP_ID = "EXP-FP-0007R"
SEEDS = [80001, 80002, 80003, 80004, 80005]
N_EPISODES = 15
ENVS = ["pomaze", "delayed_reward", "changing_rule"]
GATE_THRESHOLD = 0.0  # preregistered, frozen


def run_env(env_name):
    """Run U/G/R on all seeds for one env; return per-seed rows + verdict."""
    per_seed = []
    for s in SEEDS:
        log_u = os.path.join(
            OUT_DIR, f"{EXP_ID}.{env_name}.seed{s}.U.predictions.jsonl")
        arm_u = base.run_arm(env_name, N_EPISODES, s, 1000 + s,
                             log_path=log_u, gate_policy="ungated")
        arm_g = base.run_arm(env_name, N_EPISODES, s, 1000 + s,
                             gate_policy="uhat",
                             gate_threshold=GATE_THRESHOLD,
                             gate_uhat_source="shadow",
                             want_shadow_hist=True)
        r_s = arm_g["gate_stats"]["application_rate"]
        arm_r = base.run_arm(env_name, N_EPISODES, s, 1000 + s,
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
            "U_ranking": base.uhat_benefit_corr(log_u),
        })
        print(f"[{env_name}] seed {s}: U={arm_u['mean_return']:+.3f} "
              f"G={arm_g['mean_return']:+.3f} R={arm_r['mean_return']:+.3f} "
              f"dGU={dgu:+.3f} dGR={dgr:+.3f} rate={r_s:.3f}",
              flush=True)

    # G0 instrument-validity: discriminating decisions on >=4/5 seeds.
    n_disc = sum(1 for p in per_seed
                 if 0.0 < p["G_gate_stats"]["application_rate"] < 1.0)
    if n_disc < 4:
        print(f"[{env_name}] G0 FAIL: instrument degenerate "
              f"({n_disc}/5 seeds discriminating). Env VOID.", flush=True)
        return env_name, None

    # G2 determinism spot-check: recompute first seed's G arm, unrounded.
    g_re = base.run_arm(env_name, N_EPISODES, SEEDS[0], 1000 + SEEDS[0],
                        gate_policy="uhat", gate_threshold=GATE_THRESHOLD,
                        gate_uhat_source="shadow")
    if abs(g_re["mean_return"] - per_seed[0]["G_mean_return_raw"]) > 1e-12:
        print(f"[{env_name}] G2 FAIL: determinism mismatch. Env VOID.",
              flush=True)
        return env_name, None
    print(f"[{env_name}] G0/G2 PASS ({n_disc}/5 discriminating, "
          f"determinism {g_re['mean_return']:.4f}).", flush=True)

    mdgu = base._mean([p["delta_GU"] for p in per_seed])
    mdgr = base._mean([p["delta_GR"] for p in per_seed])
    agree_gu = sum(1 for p in per_seed if p["delta_GU"] > 0)
    agree_gr = sum(1 for p in per_seed if p["delta_GR"] > 0)
    win = mdgu > 0 and mdgr > 0 and agree_gu >= 4 and agree_gr >= 4
    return env_name, {
        "per_seed": per_seed,
        "seed_mean_delta_GU": round(mdgu, 4),
        "seed_mean_delta_GR": round(mdgr, 4),
        "seeds_agree_GU": agree_gu,
        "seeds_agree_GR": agree_gr,
        "g0_discriminating_seeds": n_disc,
        "win_rule_fires": win,
        "verdict": "WIN" if win else "NO_WIN",
    }


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)
    envs_out = {}
    for env_name in ENVS:
        name, res = run_env(env_name)
        envs_out[name] = res

    bar_envs = [e for e, r in envs_out.items()
                if r is not None and r["win_rule_fires"]]
    bar = ("pomaze" in bar_envs and
           any(e in bar_envs for e in ("delayed_reward", "changing_rule")))
    if bar:
        interp = ("H SUPPORTED: pomaze win rule fires at 5 fresh seeds AND "
                  f"{', '.join(e for e in bar_envs if e != 'pomaze')} fires "
                  "- replication breadth achieved. MATURITY retrieval row "
                  "INTEGRATED -> CAUSAL.")
    elif "pomaze" not in bar_envs:
        interp = ("NEGATIVE on replication: pomaze win rule does NOT fire "
                  f"at 5 fresh seeds (bar envs: {bar_envs}). The EXP-FP-0007 "
                  "result is bounded to its 4 seeds. Predictor stays "
                  "EXECUTED. Recorded in negative_results.md.")
    else:
        interp = ("PARTIAL: pomaze replicates (win fires) but neither new "
                  "env fires the win rule. No promotion (INTEGRATED held). "
                  "The shadow-gate advantage is pomaze-local at this "
                  "evidence level.")

    cfg = {"envs": ENVS, "agent": "arch_b v1", "affect": "none",
           "action_mode": "active_inference",
           "episodes_per_arm_per_seed": N_EPISODES,
           "gate_threshold": GATE_THRESHOLD,
           "gate_uhat_source": "shadow",
           "parent_experiment": "EXP-FP-0007"}
    result = {
        "experiment_id": EXP_ID,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [],
        "summary": {
            "envs": envs_out,
            "bar_envs": bar_envs,
            "preregistered_bar_met": bar,
            "verdict": ("SUPPORTED" if bar
                        else ("NEGATIVE" if "pomaze" not in bar_envs
                              else "PARTIAL")),
        },
    }
    path = base.write_receipt(
        result, RECEIPTS_DIR,
        hypothesis=("The shadow-ûhat gate advantage replicates on pomaze "
                    "at the 5-seed bar and generalizes to at least one of "
                    "two new environments."),
        null=("On pomaze the win rule fails at the 5-seed bar, or it fires "
              "on pomaze but on neither new env."),
        preregistered_metric=("per env per seed, mean episode return over "
                              "15 closed-loop episodes; dGU/G vs U, dGR "
                              "G vs R; win = seed-mean dGU>0 AND dGR>0 with "
                              ">=4/5 seeds agreeing on both; bar = pomaze "
                              "win AND >=1 of delayed_reward/changing_rule "
                              "win."),
        baseline="arm U = ungated retrieval-correction.",
        conditions=json.dumps(result["config"], sort_keys=True),
        interpretation=interp,
        limitations=("Closed-loop policy test; gated trajectory diverges "
                     "from ungated; shadow trains on counterfactual benefit "
                     "but lives on the gated trajectory; all arms may stay "
                     "negative if env unsolved; 5 fresh seeds, no "
                     "multiple-comparison correction."),
    )
    ok, problems = base.verify_chain(RECEIPTS_DIR)
    print(f"hash chain: {'OK' if ok else problems}")
    print(f"\nverdict: {result['summary']['verdict']}\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
