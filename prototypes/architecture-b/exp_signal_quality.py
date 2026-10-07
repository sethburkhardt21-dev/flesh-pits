"""EXP-FP-0009 — env-dependence characterization of the shadow-trained uhat
ranking signal (mechanism lead for B gap #2).

Preregistered: experiments/preregistration_SIGNAL_QUALITY.json (sealed
before any code was written; this file is the executor, not the spec).

CHARACTERIZATION, not a kill: no policy comparison, no win/lose rule.
One characterization arm per (env, seed): ungated correction stream
(correction applied whenever available) with gate_uhat_source="shadow",
so the shadow UsefulnessPredictor trains online on unconditional benefit.
Driver-side instrumentation ONLY: wraps agent.shadow_usefulness.update to
record (qfeat, uhat_pre_update, benefit) per correction-available tick.
predictions.py, agent.py, experiments.py are NOT modified.

Per-seed measurements (preregistered):
  (a) corr(uhat_pre, benefit)
  (b) benefit base rate P(benefit>0), P(benefit>=1e-9), mean benefit,
      sign-agreement P((uhat_pre>0)==(benefit>0))
  (c) sign vs magnitude: corr(uhat,sign(benefit)), corr(uhat,|benefit|),
      corr within benefit>0 / benefit<=0 strata, benefit quantiles
  (d) feature carriers: per-feature mean/std, corr(x_j, benefit), final
      shadow weights w/b
  (e) gate application rate at threshold 0: fraction of correction-available
      ticks with online uhat_pre > 0

Frozen gates: G1 tripwire CLEAN pre-run (checked by operator);
G2 determinism spot-check per env (first seed recomputed; unrounded mean
benefit and corr match at 1e-12); G3 hash-chained lane receipt + detail copy.

Usage:
    python3 exp_signal_quality.py
Receipt: flesh-pits/receipts/EXP-FP-0009.json (hash-chained).
Detail: prototypes/architecture-b/receipts/EXP-FP-0009.json (pre-chain copy).
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

import exp_retrieval_gate_shadow as base  # noqa: E402 — corr/_mean, verbatim
from env_interface import derive_seed, config_hash  # noqa: E402
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")
DETAIL_DIR = os.path.join(HERE, "receipts")

EXP_ID = "EXP-FP-0009"
SEEDS = [81001, 81002, 81003, 81004, 81005]
N_EPISODES = 15
ENVS = ["pomaze", "delayed_reward", "changing_rule", "compositional_rule",
        "cue_delayed_reward"]
FEAT_NAMES = ["mean_sim", "n_nbrs_5", "uncertainty", "e1_ema"]
MIN_TICKS = 100  # S1 non-degeneracy bar per seed


def _quantile(xs, q):
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


class ShadowRecorder:
    """Driver-side instrumentation: record (qfeat, uhat_pre, benefit) at
    every shadow_usefulness.update call. Side-effect free w.r.t. the agent
    (predict is pure); the update itself is unchanged. Deterministic."""

    def __init__(self, agent):
        self.agent = agent
        self.records = []  # (qfeat tuple, uhat_pre, benefit)
        self._orig = agent.shadow_usefulness.update

    def __call__(self, qfeat, benefit):
        uhat_pre = self.agent.shadow_usefulness.predict(qfeat)
        self.records.append((tuple(qfeat), uhat_pre, benefit))
        return self._orig(qfeat, benefit)

    def install(self):
        self.agent.shadow_usefulness.update = self.__call__


def run_characterization_arm(env_name, seed):
    """One characterization arm: ungated stream, shadow trains on
    unconditional benefit. Returns per-seed measurement dict."""
    agent = make_agent(ENVS_BY_NAME[env_name], seed=1000 + seed,
                       gate_policy="ungated", gate_uhat_source="shadow")
    rec = ShadowRecorder(agent)
    rec.install()
    eps = run_closed_loop(env_name, agent, N_EPISODES, seed)
    records = rec.records
    feats = [r[0] for r in records]
    uhat = [r[1] for r in records]
    ben = [r[2] for r in records]
    n = len(records)

    feat_marg = []
    for j in range(4):
        xs = [f[j] for f in feats]
        feat_marg.append({
            "mean": base._mean(xs),
            "std": statistics.pstdev(xs) if n > 1 else 0.0,
            "corr_with_benefit": base.corr(xs, ben),
        })
    signs = [_sign(b) for b in ben]
    absben = [abs(b) for b in ben]
    pos = [(u, b) for u, b in zip(uhat, ben) if b > 0]
    nonpos = [(u, b) for u, b in zip(uhat, ben) if b <= 0]
    agree = sum(1 for u, b in zip(uhat, ben) if (u > 0) == (b > 0))
    mean_return = base._mean([e["return"] for e in eps])
    total_ticks = agent._tick
    out = {
        "seed": seed,
        "n_ticks_avail": n,
        "total_ticks": total_ticks,
        "avail_rate": n / total_ticks if total_ticks else 0.0,
        "mean_return": round(mean_return, 4),
        "mean_return_raw": mean_return,
        "mean_benefit": base._mean(ben),
        "mean_benefit_raw": base._mean(ben),
        "std_benefit": statistics.pstdev(ben) if n > 1 else 0.0,
        "p_benefit_pos": sum(1 for b in ben if b > 0) / n if n else 0.0,
        "p_benefit_nonneg": sum(1 for b in ben if b >= 1e-9) / n if n else 0.0,
        "corr_uhat_benefit": base.corr(uhat, ben),
        "corr_uhat_sign": base.corr(uhat, signs),
        "corr_uhat_absbenefit": base.corr(uhat, absben),
        "corr_uhat_benefit_pos": base.corr([u for u, _ in pos],
                                           [b for _, b in pos]),
        "corr_uhat_benefit_nonpos": base.corr([u for u, _ in nonpos],
                                              [b for _, b in nonpos]),
        "sign_agreement": agree / n if n else 0.0,
        "benefit_q": {"p10": _quantile(ben, 0.10), "p50": _quantile(ben, 0.50),
                      "p90": _quantile(ben, 0.90)},
        "absbenefit_q": {"p10": _quantile(absben, 0.10),
                         "p50": _quantile(absben, 0.50),
                         "p90": _quantile(absben, 0.90)},
        "apply_rate_t0": sum(1 for u in uhat if u > 0) / n if n else 0.0,
        "apply_rate_t0_nonneg": sum(1 for u in uhat if u >= 0) / n
        if n else 0.0,
        "feature_marginals": feat_marg,
        "shadow_w": list(agent.shadow_usefulness.w),
        "shadow_b": agent.shadow_usefulness.b,
        "degenerate": n < MIN_TICKS,
    }
    return out


def run_env(env_name):
    per_seed = []
    for s in SEEDS:
        row = run_characterization_arm(env_name, s)
        per_seed.append(row)
        print(f"[{env_name}] seed {s}: n={row['n_ticks_avail']} "
              f"corr={row['corr_uhat_benefit']} "
              f"p_pos={row['p_benefit_pos']:.3f} "
              f"apply_t0={row['apply_rate_t0']:.3f} "
              f"{'DEGENERATE' if row['degenerate'] else ''}", flush=True)

    # S1 non-degeneracy: >=4/5 seeds with >=MIN_TICKS correction ticks.
    n_ok = sum(1 for p in per_seed if not p["degenerate"])
    if n_ok < 4:
        print(f"[{env_name}] S1 FAIL: degenerate ({n_ok}/5 seeds). "
              f"Env DEGENERATE.", flush=True)
        return env_name, None

    # G2 determinism spot-check: recompute first seed, unrounded compare.
    re = run_characterization_arm(env_name, SEEDS[0])
    mb0 = per_seed[0]["mean_benefit_raw"]
    cb0 = per_seed[0]["corr_uhat_benefit"]
    if (abs(re["mean_benefit_raw"] - mb0) > 1e-12 or
            abs((re["corr_uhat_benefit"] or 0) - (cb0 or 0)) > 1e-12):
        print(f"[{env_name}] G2 FAIL: determinism mismatch. Env VOID.",
              flush=True)
        return env_name, None
    print(f"[{env_name}] S1/G2 PASS ({n_ok}/5 non-degenerate, "
          f"determinism 1e-12).", flush=True)

    ok_rows = [p for p in per_seed if not p["degenerate"]]
    seed_corrs = [p["corr_uhat_benefit"] for p in ok_rows
                  if p["corr_uhat_benefit"] is not None]
    # Carrier feature per env: argmax |seed-mean corr(x_j, benefit)|.
    carrier = None
    carrier_val = None
    for j, fname in enumerate(FEAT_NAMES):
        cs = [p["feature_marginals"][j]["corr_with_benefit"] for p in ok_rows]
        cs = [c for c in cs if c is not None]
        if cs:
            v = abs(base._mean(cs))
            if carrier_val is None or v > carrier_val:
                carrier_val = v
                carrier = fname
    # Learned-weight sign alignment with corr(x_j, benefit).
    w_align = []
    for j, fname in enumerate(FEAT_NAMES):
        cs = [p["feature_marginals"][j]["corr_with_benefit"] for p in ok_rows]
        cs = [c for c in cs if c is not None]
        ws = [p["shadow_w"][j] for p in ok_rows]
        if cs and ws:
            mean_c = base._mean(cs)
            mean_w = base._mean(ws)
            w_align.append({"feature": fname,
                            "seed_mean_corr": mean_c,
                            "seed_mean_w": mean_w,
                            "sign_aligned": (mean_c * mean_w) > 0})
    return env_name, {
        "per_seed": per_seed,
        "n_nondegenerate": n_ok,
        "seed_mean_corr": base._mean(seed_corrs) if seed_corrs else None,
        "seed_corrs": seed_corrs,
        "seed_mean_p_pos": base._mean([p["p_benefit_pos"] for p in ok_rows]),
        "seed_mean_std_benefit": base._mean(
            [p["std_benefit"] for p in ok_rows]),
        "seed_mean_absbenefit_p50": base._mean(
            [p["absbenefit_q"]["p50"] for p in ok_rows
             if p["absbenefit_q"]["p50"] is not None]),
        "seed_mean_avail_rate": base._mean(
            [p["avail_rate"] for p in ok_rows]),
        "seed_mean_apply_t0": base._mean(
            [p["apply_rate_t0"] for p in ok_rows]),
        "seed_mean_corr_sign": base._mean(
            [p["corr_uhat_sign"] for p in ok_rows
             if p["corr_uhat_sign"] is not None]),
        "seed_mean_corr_abs": base._mean(
            [p["corr_uhat_absbenefit"] for p in ok_rows
             if p["corr_uhat_absbenefit"] is not None]),
        "carrier_feature": carrier,
        "carrier_abs_corr": carrier_val,
        "weight_alignment": w_align,
    }


def _spearman(xs, ys):
    """Rank correlation over env-level aggregates (descriptive, n=5)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = base._mean(rx), base._mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return num / math.sqrt(dx * dy)


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    envs_out = {}
    for env_name in ENVS:
        name, res = run_env(env_name)
        envs_out[name] = res

    # Cross-env rank correlations (descriptive, n=5, no significance claims).
    valid = {e: r for e, r in envs_out.items() if r is not None}
    xcorr = {e: r["seed_mean_corr"] for e, r in valid.items()}
    props = {
        "p_benefit_pos": {e: r["seed_mean_p_pos"] for e, r in valid.items()},
        "std_benefit": {e: r["seed_mean_std_benefit"]
                        for e, r in valid.items()},
        "absbenefit_p50": {e: r["seed_mean_absbenefit_p50"]
                           for e, r in valid.items()},
        "avail_rate": {e: r["seed_mean_avail_rate"] for e, r in valid.items()},
        "apply_rate_t0": {e: r["seed_mean_apply_t0"]
                          for e, r in valid.items()},
        "corr_sign": {e: r["seed_mean_corr_sign"] for e, r in valid.items()},
        "corr_abs": {e: r["seed_mean_corr_abs"] for e, r in valid.items()},
    }
    envs_sorted = sorted(valid.keys())
    rank_corrs = {}
    for pname, pvals in props.items():
        xs = [xcorr[e] for e in envs_sorted]
        ys = [pvals[e] for e in envs_sorted]
        if any(v is None for v in xs + ys):
            rank_corrs[pname] = None
        else:
            rank_corrs[pname] = _spearman(xs, ys)

    rank_order = sorted([(e, r["seed_mean_corr"]) for e, r in valid.items()
                       if r["seed_mean_corr"] is not None],
                      key=lambda t: t[1], reverse=True)

    degenerate_envs = [e for e, r in envs_out.items() if r is None]
    s1_ok = len(degenerate_envs) == 0
    interp = (f"EXP-FP-0009 characterization complete: {len(valid)}/5 envs "
              f"non-degenerate. Signal-quality rank_order (seed-mean "
              f"corr(uhat,benefit)): "
              + ", ".join(f"{e}={c:+.3f}" for e, c in rank_order)
              + ". Carrier features: "
              + ", ".join(f"{e}:{r['carrier_feature']}"
                          for e, r in valid.items())
              + ". Cross-env rank correlations with signal quality: "
              + ", ".join(f"{k}={v:+.2f}" if v is not None else f"{k}=n/a"
                          for k, v in rank_corrs.items())
              + (f". DEGENERATE envs: {degenerate_envs}." if degenerate_envs
                 else ""))
    limitations = ("Observational characterization (no policy intervention); "
                   "shadow predictor trains on the ungated trajectory; "
                   "n=5 envs for cross-env rank correlations (no "
                   "significance claims); 5 fresh seeds per env; 15 "
                   "episodes per arm. CONSCIOUSNESS: UNRESOLVED.")

    cfg = {"envs": ENVS, "agent": "arch_b v1", "affect": "none",
           "action_mode": "active_inference",
           "episodes_per_arm_per_seed": N_EPISODES,
           "gate_policy": "ungated", "gate_uhat_source": "shadow",
           "parent_experiment": "EXP-FP-0007R",
           "preregistration": "experiments/preregistration_SIGNAL_QUALITY.json"}
    result = {
        "experiment_id": EXP_ID,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [],
        "summary": {
            "envs": envs_out,
            "signal_quality_ranking": rank_order,
            "cross_env_rank_correlations": rank_corrs,
            "degenerate_envs": degenerate_envs,
            "s1_all_nondegenerate": s1_ok,
            "min_ticks_per_seed": MIN_TICKS,
            "verdict": "CHARACTERIZED" if s1_ok else "PARTIAL (degenerate)",
        },
    }
    os.makedirs(DETAIL_DIR, exist_ok=True)
    detail_path = os.path.join(DETAIL_DIR, f"{EXP_ID}.json")
    with open(detail_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"detail copy: {detail_path}")

    path = write_receipt(
        result, RECEIPTS_DIR,
        hypothesis=("The shadow-trained uhat ranking-signal quality varies "
                    "systematically across envs, predictable from "
                    "env-measurable properties (base rate, magnitude "
                    "distribution, feature marginals, carrier feature)."),
        null=("No systematic env-dependence identifiable; the "
              "pomaze/changing_rule contrast is noise."),
        preregistered_metric=("per (env,seed): corr(uhat_pre,benefit); "
                              "P(benefit>0); sign/magnitude decomposition; "
                              "feature marginals + carrier; gate apply rate "
                              "at threshold 0; cross-env rank correlations. "
                              "Descriptive success S1-S4; no policy "
                              "win/lose rule."),
        baseline=("ungated unconditional-correction stream; shadow predictor "
                  "trains online on unconditional benefit (characterization "
                  "arm, no gating intervention)."),
        conditions=json.dumps(result["config"], sort_keys=True),
        interpretation=interp,
        limitations=limitations,
    )
    ok, problems = verify_chain(RECEIPTS_DIR)
    print(f"hash chain: {'OK' if ok else problems}")
    print(f"\nverdict: {result['summary']['verdict']}\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())