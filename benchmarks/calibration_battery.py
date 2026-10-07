"""Calibration battery — EXP-FP-CALIB-01 (§9 item 9).

Runs Architecture B closed-loop on changing_rule (5 fresh seeds), consumes
the §30 PredictionLog JSONL triples (prediction, confidence, uncertainty,
actual) — never self-report — and scores B's stated uncertainties as
probabilistic claims:

  D1 next_obs:            o=1(mean|e0| <= EPS_OBS),   p = erf(EPS_OBS/(s*sqrt2)), s=ounc
  D2 action_consequence:  o=1(|rerr|   <= EPS_RW),    p = erf(EPS_RW/(s*sqrt2)),  s=runc
  D3 competence_failure:  o=1(mean|e0| >  TAU_FAIL),  p = erfc(TAU_FAIL/(s*sqrt2)), s=ounc
  D4 retrieval_usefulness: o=1(benefit > 0),          p = Phi(uhat/s_u)

Per-domain Brier score vs the sequential-Laplace climatology baseline,
ECE/MCE reliability, signed calibration error, and the preregistered
decision tree (see benchmarks/calibration_battery_preregistration.json).

Deterministic. Stdlib only.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB_DIR = os.path.join(HERE, "..", "prototypes", "architecture-b")
FP_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(AB_DIR))
sys.path.insert(0, os.path.join(os.path.abspath(FP_DIR), "experiments"))

from env_interface import config_hash  # noqa: E402
from experiments import (  # noqa: E402
    ENVS_BY_NAME,
    make_agent,
    run_closed_loop,
)

# -- preregistered constants (frozen 2026-10-07, BEFORE any battery run) ------
EPS_OBS = 0.2682    # hist median mean_abs_error, 2340 published ticks
EPS_RW = 0.2067     # hist median |rerr|, same ticks
TAU_FAIL = 0.4156   # hist p90 mean_abs_error, same ticks
SEEDS = [73101, 73102, 73103, 73104, 73105]
N_EPISODES = 10
N_BINS = 10
OUT_DIR = os.path.join(os.path.abspath(AB_DIR), "experiments_out")


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clamp01(p: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, p))


def brier(p, o):
    ps = [_clamp01(x) for x in p]
    return sum((a - b) ** 2 for a, b in zip(ps, o)) / len(o)


def climatology_brier(o):
    # Sequential Laplace-smoothed running base rate: only past outcomes.
    a = 1.0
    n = 0
    err = 0.0
    for oi in o:
        p = a / (n + 2.0)
        err += (p - oi) ** 2
        a += oi
        n += 1
    return err / len(o)


def reliability(o, p, n_bins=N_BINS):
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, pi in enumerate(p)
               if (lo <= pi < hi) or (b == n_bins - 1 and pi == hi)]
        if not idx:
            bins.append({"bin": [lo, hi], "n": 0,
                         "mean_p": None, "obs_rate": None})
            continue
        mean_p = sum(p[i] for i in idx) / len(idx)
        obs = sum(o[i] for i in idx) / len(idx)
        bins.append({"bin": [lo, hi], "n": len(idx),
                     "mean_p": mean_p, "obs_rate": obs})
    filled = [x for x in bins if x["n"] > 0]
    ece = sum(abs(x["obs_rate"] - x["mean_p"]) * x["n"] for x in filled) / len(o)
    mce = max(abs(x["obs_rate"] - x["mean_p"]) for x in filled) if filled else 1.0
    return bins, ece, mce


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)
                    * sum((y - my) ** 2 for y in ys))
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def score_domain(name, triples):
    """triples: list of (p, o, sigma, err). Returns scored dict + verdict."""
    n = len(triples)
    diag = {"n": n}
    if n == 0:
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": "zero triples collected"})
        return diag
    p = [t[0] for t in triples]
    o = [t[1] for t in triples]
    sig = [t[2] for t in triples]
    err = [t[3] for t in triples]
    if any(not math.isfinite(v) for v in p + o + sig):
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": "non-finite p/o/sigma"})
        return diag
    sd = (sum((x - sum(p) / n) ** 2 for x in p) / n) ** 0.5 if n > 1 else 0.0
    if sd < 1e-9:
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": "degenerate probabilities (std < 1e-9)"})
        return diag
    base_rate = sum(o) / n
    if not (0.02 <= base_rate <= 0.98):
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": f"degenerate base rate {base_rate:.4f}"})
        return diag
    b_b = brier(p, o)
    b_c = climatology_brier(o)
    b_half = brier([0.5] * n, o)
    bins, ece, mce = reliability(o, p)
    s = sum(pi - oi for pi, oi in zip(p, o)) / n
    corr = pearson(sig, err)
    verdict = None
    if b_b >= b_c:
        direction = ("over-confident" if s > 0.05
                     else "under-confident" if s < -0.05
                     else "dispersion error")
        verdict = f"MISCALIBRATED (fails to beat climatology; {direction})"
    elif ece <= 0.10 and mce <= 0.25:
        verdict = "CALIBRATED"
    else:
        direction = ("over-confident" if s > 0.05
                     else "under-confident" if s < -0.05
                     else "dispersion error")
        verdict = (f"MISCALIBRATED (beats climatology but reliability "
                   f"off-diagonal; {direction})")
    diag.update({
        "base_rate": base_rate, "brier_B": b_b, "brier_climo": b_c,
        "brier_const05": b_half, "skill_vs_climo": 1.0 - b_b / b_c,
        "ece": ece, "mce": mce, "signed_error": s,
        "corr_sigma_err": corr, "verdict": verdict,
        "reliability_bins": bins,
    })
    return diag


def collect(log_path):
    """Read §30 JSONL -> per-domain (p, o, sigma, err) triples."""
    d = {"D1": [], "D2": [], "D3": [], "D4": []}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            t = r["target"]
            if t == "next_obs":
                s = r["uncertainty"]
                e = r["mean_abs_error"]
                d["D1"].append((_clamp01(math.erf(EPS_OBS / (s * math.sqrt(2.0))))
                                if s > 0 else 0.5,
                                1.0 if e <= EPS_OBS else 0.0, s, e))
                d["D3"].append((_clamp01(math.erfc(TAU_FAIL / (s * math.sqrt(2.0))))
                                if s > 0 else 0.5,
                                1.0 if e > TAU_FAIL else 0.0, s, e))
            elif t == "action_consequence":
                s = r["uncertainty"]
                e = r["abs_error"]
                d["D2"].append((_clamp01(math.erf(EPS_RW / (s * math.sqrt(2.0))))
                                if s > 0 else 0.5,
                                1.0 if e <= EPS_RW else 0.0, s, e))
            elif t == "retrieval_usefulness":
                mu = r["prediction"]
                s = r["uncertainty"]
                b = r["actual"]
                e = r["abs_error"]
                d["D4"].append((_clamp01(_phi(mu / s)) if s > 0 else 0.5,
                                1.0 if b > 0 else 0.0, s, abs(b)))
    return d


def run_seed(seed, out_dir):
    logp = os.path.join(out_dir, f"EXP-FP-CALIB-01.seed{seed}.predictions.jsonl")
    agent = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                       log_path=logp, affect="none")
    eps = run_closed_loop("changing_rule", agent, N_EPISODES, seed)
    if agent.plog:
        agent.plog.close()
    return logp, eps


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EXP-FP-CALIB-01 calibration battery")
    ap.add_argument("--out", default=None,
                    help="results JSON path (default: benchmarks/calibration_battery_results.json)")
    ap.add_argument("--run", action="store_true",
                    help="execute the closed-loop runs (seeds 73101-73105)")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="existing prediction JSONL logs to score instead of running")
    args = ap.parse_args(argv)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    logs = []
    episodes = {}
    if args.logs:
        logs = args.logs
    elif args.run:
        for seed in SEEDS:
            print(f"--- seed {seed} ---", flush=True)
            logp, eps = run_seed(seed, OUT_DIR)
            logs.append(logp)
            episodes[str(seed)] = eps
    else:
        raise SystemExit("pass --run or --logs")

    pooled = {"D1": [], "D2": [], "D3": [], "D4": []}
    for logp in logs:
        c = collect(logp)
        for k in pooled:
            pooled[k].extend(c[k])

    names = {"D1": "next_obs", "D2": "action_consequence",
             "D3": "competence_failure", "D4": "retrieval_usefulness"}
    domains = {}
    for k in ("D1", "D2", "D3", "D4"):
        sc = score_domain(names[k], pooled[k])
        domains[names[k]] = sc
        print(f"{k} {names[k]}: n={sc['n']} verdict={sc.get('verdict')} "
              f"brier_B={sc.get('brier_B')} brier_climo={sc.get('brier_climo')} "
              f"ECE={sc.get('ece')} S={sc.get('signed_error')}")

    cfg = {"experiment_id": "EXP-FP-CALIB-01",
           "contract_version": "1.0.0",
           "env": "changing_rule", "env_version": "1.0.0",
           "agent": "arch_b", "affect": "none",
           "episodes_per_seed": N_EPISODES, "primary_seeds": SEEDS,
           "thresholds": {"EPS_OBS": EPS_OBS, "EPS_RW": EPS_RW,
                          "TAU_FAIL": TAU_FAIL},
           "bins": N_BINS}
    results = {
        "experiment_id": "EXP-FP-CALIB-01",
        "started_utc": started,
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "seeds": SEEDS,
        "logs": logs,
        "domains": domains,
    }
    outp = args.out or os.path.join(HERE, "calibration_battery_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"results -> {outp}")
    return results, cfg, started


if __name__ == "__main__":
    main()
