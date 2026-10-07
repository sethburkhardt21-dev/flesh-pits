"""EXP-FP-0040 - retrieval-gate SELECTIVE deficit on pomaze.

Preregistered spec (sealed in experiments/EXPERIMENT_REGISTRY.md before any
code was written, and in experiments/preregistration_EXP-FP-0040.json; this
file is the executor, not the spec).

Arms (paired by (seed, episode index); identical primary seeds):
  U = gate_policy="ungated" (correction whenever available; current default)
  G = gate_policy="uhat", gate_threshold=0.0 FROZEN, gate_uhat_source="shadow"
      (EXP-FP-0007 instrument reused VERBATIM)
  N = gate_policy="uhat", gate_threshold=+inf, gate_uhat_source="shadow"
      (never-apply no-correction arm; store intact)

Metric (EXACT): dUG_s = mean_s^U - mean_s^G; dGN_s = mean_s^G - mean_s^N;
dUN_s = mean_s^U - mean_s^N (means over 15 episodes). Trial-level:
L_e,s = mean realized benefit over correction-available ticks of episode e
from the U-arm PredictionLog; De,s = R_e,s^U - R_e,s^G; corr_s = pearson
over 15 episodes.

Verdict rule (priority order): CEILING (seed-mean |dUN| <= 0.25) ->
UNIFORM (seed-mean dGN <= 0.25 AND seed-mean dUG > 0.25) ->
SELECTIVE (seed-mean dGN > 0.25 AND dUG/dUN < 0.5 AND >=3/4 seeds
sign(dGN)>0 AND >=3/4 seeds corr_s>0) -> NO-DEFICIT (seed-mean dUG <= 0).

Frozen gates: G0 instrument-validity (G rate in (0,1) on >=3/4 seeds),
G1 fabrication-tripwire CLEAN pre-run, G2 determinism spot-check
(first-seed G arm recomputed, unrounded 1e-12), G3 hash-chained receipt.

Additive only: imports run-arm machinery from exp_retrieval_gate_shadow;
retrieval_gate.py, predictions.py, agent.py untouched. No random.* usage.

Usage:
    python3 exp_retrieval_gate_selective.py
Receipt: flesh-pits/receipts/EXP-FP-0040.json (hash-chained).
"""

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
    CONTRACT_VERSION, config_hash,
)
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402
from exp_retrieval_gate_shadow import (  # noqa: E402
    corr, uhat_benefit_corr,
)

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")
OUT_DIR = os.path.join(HERE, "experiments_out")

EXP_ID = "EXP-FP-0040"
SEEDS = [90101, 90102, 90103, 90104]
N_EPISODES = 15
ENV_NAME = "pomaze"
GATE_THRESHOLD = 0.0  # preregistered, frozen

PREREG = {
    "hypothesis": ("The shadow-uhat gate is a faithful selective filter: "
                   "gated retrieval performs between unconditional and "
                   "no-correction, with the deficit concentrated on "
                   "episodes where retrieval mattered - not a uniform "
                   "collapse."),
    "null": ("CEILING: |mean(U)-mean(N)| <= 0.25 seed-mean (correction "
             "carries no return leverage); UNIFORM: gated ~= no-correction "
             "while clearly below unconditional (broken filter); "
             "NO-DEFICIT: gated >= unconditional (the 0007/0007R direction)."),
    "metric": ("per seed: dUG = mean(U)-mean(G), dGN = mean(G)-mean(N), "
               "dUN = mean(U)-mean(N) over 15 paired episodes; trial-level "
               "corr_s = pearson(De, L_e) with L_e = mean realized benefit "
               "per episode from the U-arm PredictionLog; verdict priority "
               "CEILING -> UNIFORM -> SELECTIVE -> NO-DEFICIT."),
    "baseline": "arm U = ungated (current ArchB default).",
    "arms": ("U: gate_policy='ungated'; G: gate_policy='uhat', "
             "threshold=0.0 frozen, gate_uhat_source='shadow' (0007 "
             "instrument verbatim); N: gate_policy='uhat', "
             "gate_threshold=+inf, gate_uhat_source='shadow' "
             "(never-apply no-correction arm; store intact)."),
    "procedure": ("pomaze v1.0.0, arch_b v1, affect='none', "
                  "action_mode='active_inference', 15 episodes, identical "
                  "primary seeds across arms (paired)."),
    "seeds": list(SEEDS),
    "conditions": {"env": "pomaze v1.0.0", "agent": "arch_b v1",
                   "affect": "none", "action_mode": "active_inference",
                   "episodes": N_EPISODES, "gate_threshold": GATE_THRESHOLD,
                   "gate_uhat_source": "shadow", "N_threshold": "+inf",
                   "contract": CONTRACT_VERSION},
}


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def run_arm_full(env_name, n_eps, primary_seed, agent_seed, log_path=None,
                 want_shadow_hist=False, **gate_kwargs):
    """Run one arm; keep UNROUNDED per-episode returns (needed for the
    trial-level deficit correlation). Mirrors exp_retrieval_gate_shadow's
    run_arm, extended to keep raw returns and shadow history."""
    agent = make_agent(ENVS_BY_NAME[env_name], seed=agent_seed,
                       log_path=log_path, **gate_kwargs)
    eps = run_closed_loop(env_name, agent, n_eps, primary_seed)
    returns_raw = [e["return"] for e in eps]
    out = {
        "mean_return_raw": _mean(returns_raw),
        "returns_raw": returns_raw,
        "gate_stats": agent.gate.stats(),
    }
    if want_shadow_hist:
        uh = [u for u, b in agent._shadow_hist]
        bh = [b for u, b in agent._shadow_hist]
        out["shadow_corr"] = {"n": len(uh),
                              "corr_uhat_benefit": corr(uh, bh)}
    return out


def episode_leverage(log_path):
    """Per-episode mean realized retrieval benefit from a U-arm
    PredictionLog (target='retrieval_usefulness', field 'actual').
    Returns {episode: mean_benefit}."""
    by_ep = {}
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("target") == "retrieval_usefulness":
                by_ep.setdefault(rec["episode"], []).append(rec["actual"])
    return {ep: _mean(vals) for ep, vals in by_ep.items()}


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed = []
    for s in SEEDS:
        log_u = os.path.join(OUT_DIR, f"{EXP_ID}.seed{s}.U.predictions.jsonl")
        arm_u = run_arm_full(ENV_NAME, N_EPISODES, s, 1000 + s,
                             log_path=log_u, gate_policy="ungated")
        arm_g = run_arm_full(ENV_NAME, N_EPISODES, s, 1000 + s,
                             gate_policy="uhat", gate_threshold=GATE_THRESHOLD,
                             gate_uhat_source="shadow", want_shadow_hist=True)
        arm_n = run_arm_full(ENV_NAME, N_EPISODES, s, 1000 + s,
                             gate_policy="uhat", gate_threshold=float("inf"),
                             gate_uhat_source="shadow", want_shadow_hist=True)
        u_mean = arm_u["mean_return_raw"]
        g_mean = arm_g["mean_return_raw"]
        n_mean = arm_n["mean_return_raw"]
        d_ug = u_mean - g_mean
        d_gn = g_mean - n_mean
        d_un = u_mean - n_mean
        lev = episode_leverage(log_u)
        de = [u - g for u, g in zip(arm_u["returns_raw"],
                                    arm_g["returns_raw"])]
        lvec = [lev.get(e, 0.0) for e in range(N_EPISODES)]
        corr_dl = corr(de, lvec)
        per_seed.append({
            "seed": s,
            "U_mean_return": round(u_mean, 4),
            "G_mean_return": round(g_mean, 4),
            "G_mean_return_raw": g_mean,
            "N_mean_return": round(n_mean, 4),
            "d_UG": round(d_ug, 4),
            "d_GN": round(d_gn, 4),
            "d_UN": round(d_un, 4),
            "corr_deficit_leverage": corr_dl,
            "G_gate_stats": arm_g["gate_stats"],
            "G_shadow_corr": arm_g["shadow_corr"],
            "N_gate_stats": {
                "n_available": arm_n["gate_stats"]["n_available"],
                "n_applied": arm_n["gate_stats"]["n_applied"],
                "application_rate": arm_n["gate_stats"]["application_rate"],
            },
            "N_shadow_corr": arm_n["shadow_corr"],
            "U_ranking": uhat_benefit_corr(log_u),
        })
        print(f"seed {s}: U={u_mean:+.3f} G={g_mean:+.3f} N={n_mean:+.3f} "
              f"dUG={d_ug:+.3f} dGN={d_gn:+.3f} dUN={d_un:+.3f} "
              f"corr(d,L)={corr_dl} "
              f"G_rate={arm_g['gate_stats']['application_rate']:.3f} "
              f"N_applied={arm_n['gate_stats']['n_applied']}",
              flush=True)

    # G0 instrument-validity: discriminating G decisions on >=3/4 seeds.
    n_discriminating = sum(
        1 for p in per_seed
        if 0.0 < p["G_gate_stats"]["application_rate"] < 1.0)
    if n_discriminating < 3:
        print(f"\nG0 FAIL: instrument degenerate "
              f"({n_discriminating}/4 seeds discriminating). VOID.")
        return 2

    # G2 determinism spot-check: recompute first-seed G arm, unrounded.
    g_re = run_arm_full(ENV_NAME, N_EPISODES, SEEDS[0], 1000 + SEEDS[0],
                        gate_policy="uhat", gate_threshold=GATE_THRESHOLD,
                        gate_uhat_source="shadow")
    if abs(g_re["mean_return_raw"] - per_seed[0]["G_mean_return_raw"]) > 1e-12:
        print("\nG2 FAIL: determinism spot-check mismatch. VOID.")
        return 2
    print(f"G2 determinism spot-check PASS "
          f"({g_re['mean_return_raw']:.6f} == "
          f"{per_seed[0]['G_mean_return_raw']:.6f})")

    sm_dug = _mean([p["d_UG"] for p in per_seed])
    sm_dgn = _mean([p["d_GN"] for p in per_seed])
    sm_dun = _mean([p["d_UN"] for p in per_seed])
    agree_gn = sum(1 for p in per_seed if p["d_GN"] > 0)
    agree_corr = sum(1 for p in per_seed
                     if (p["corr_deficit_leverage"] or 0.0) > 0)

    if abs(sm_dun) <= 0.25:
        verdict = "CEILING"
        interp = (f"CEILING: seed-mean |dUN| = {abs(sm_dun):.4f} <= 0.25 - "
                  f"the retrieval correction carries no return leverage on "
                  f"pomaze in this protocol; the deficit question is "
                  f"vacuous. Consistent with EXP-FP-0006 secondary. "
                  f"dUG={sm_dug:+.4f}, dGN={sm_dgn:+.4f}.")
    elif sm_dgn <= 0.25 and sm_dug > 0.25:
        verdict = "UNIFORM (gate is a broken filter)"
        interp = (f"UNIFORM: seed-mean dGN={sm_dgn:+.4f} <= 0.25 while "
                  f"seed-mean dUG={sm_dug:+.4f} > 0.25 - gated behaves like "
                  f"no-correction and sits clearly below unconditional: "
                  f"the gate is a broken filter, not a selective one. "
                  f"Mechanism: check the U-arm corr(uhat,benefit) - if it "
                  f"replicates ~+0.4 the ranking signal exists and the "
                  f"failure is in conversion at the margin, not in the "
                  f"signal; else uhat is uninformative here.")
    elif (sm_dgn > 0.25 and abs(sm_dun) > 0.25
          and sm_dug / sm_dun < 0.5 and agree_gn >= 3 and agree_corr >= 3):
        verdict = "SELECTIVE (H SUPPORTED)"
        interp = (f"SELECTIVE: seed-mean dGN={sm_dgn:+.4f} > 0.25 with "
                  f"dUG/dUN={sm_dug/sm_dun:+.3f} < 0.5 - the gated arm "
                  f"captures more than half the unconditional-vs-none gap; "
                  f"sign(dGN)>0 on {agree_gn}/4 seeds and corr(deficit, "
                  f"leverage)>0 on {agree_corr}/4 seeds: the deficit is "
                  f"concentrated where retrieval mattered, not a uniform "
                  f"collapse. The gate is a faithful selective filter.")
    elif sm_dug <= 0:
        verdict = "NO-DEFICIT"
        interp = (f"NO-DEFICIT: seed-mean dUG={sm_dug:+.4f} <= 0 - gated is "
                  f"at or above unconditional; no selective deficit because "
                  f"gating does not sacrifice retrieval value. Consistent "
                  f"with the EXP-FP-0007/0007R combined direction "
                  f"(9-seed dGU ~ +0.09). dGN={sm_dgn:+.4f}, "
                  f"dUN={sm_dun:+.4f}.")
    else:
        verdict = "MIXED"
        interp = (f"MIXED: dUG={sm_dug:+.4f}, dGN={sm_dgn:+.4f}, "
                  f"dUN={sm_dun:+.4f}; sign(dGN)>0 on {agree_gn}/4 seeds, "
                  f"corr(deficit,leverage)>0 on {agree_corr}/4 seeds. No "
                  f"preregistered branch fires exactly.")

    result = {
        "experiment_id": EXP_ID,
        "config": PREREG["conditions"],
        "config_hash": config_hash(PREREG["conditions"]),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [],
        "summary": {
            "per_seed": per_seed,
            "seed_mean_d_UG": round(sm_dug, 4),
            "seed_mean_d_GN": round(sm_dgn, 4),
            "seed_mean_d_UN": round(sm_dun, 4),
            "seeds_sign_dGN_positive": agree_gn,
            "seeds_corr_deficit_leverage_positive": agree_corr,
            "g0_discriminating_seeds": n_discriminating,
            "verdict": verdict,
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
                     "the world model; closed-loop policy test - arms "
                     "diverge from shared initial conditions; episode "
                     "pairing by index shares initial conditions, not "
                     "trajectories; arm N is no-CORRECTION (gate blocks "
                     "100%), distinct from M1's store-disabled lesion; "
                     "the +inf threshold is an additive-only "
                     "never-apply encoding, not a new policy class."),
    )
    ok, problems = verify_chain(RECEIPTS_DIR)
    print(f"hash chain: {'OK' if ok else problems}")
    print(f"\nverdict: {verdict}\n  {path}")
    print(f"seed-mean dUG={sm_dug:+.4f} dGN={sm_dgn:+.4f} dUN={sm_dun:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
