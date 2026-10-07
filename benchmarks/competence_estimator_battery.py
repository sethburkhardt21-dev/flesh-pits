"""Competence-estimator battery — EXP-FP-0070 (variant A).

Runs Architecture B closed-loop on changing_rule (4 fresh seeds), then
replays each seed's §30 PredictionLog JSONL stream CAUSALLY through the
online CompetenceEstimator (prequential: predict tick t from history < t +
tick-t prediction-time fields, then train on tick t's outcome).

Per-domain scoring mirrors EXP-FP-CALIB-01 (same binary events, same frozen
thresholds, same sequential-climatology baseline, same ECE/MCE gates), plus:
  - corr(predicted_uncertainty, |actual_error|)  (preregistered >= 0.40)
  - corr(ehat, |actual_error|)                   (expected-error coupling)
Kill arms (preregistered): anchor-only (learn=False) must reproduce
Brier_climo to 1e-9; error-permutation decomposition intact<permuted<climo.

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

from calibration_battery import (  # noqa: E402
    EPS_OBS, EPS_RW, TAU_FAIL, N_EPISODES, N_BINS,
    brier, climatology_brier, reliability, pearson,
)
from env_interface import config_hash  # noqa: E402
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from competence_estimator import CompetenceEstimator  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

SEEDS = [73201, 73202, 73203, 73204]
OUT_DIR = os.path.join(os.path.abspath(AB_DIR), "experiments_out")
DOMAINS = ("next_obs", "action_consequence",
           "competence_failure", "retrieval_usefulness")
# predicted_uncertainty mapping per domain for the coupling target
U_IS_P = {"competence_failure"}  # u = p (failure prob); else u = 1 - p


def _clamp01(p):
    return min(1.0 - 1e-9, max(1e-9, p))


def collect_ticks(log_path):
    """JSONL -> ordered list of per-tick dicts.

    Each tick: ctx (prediction-time fields, known BEFORE the outcome) and
    out (realized outcomes). update_norm is lagged by one tick in replay.
    """
    by_tick = {}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            t = (r["tick"], r["episode"])
            d = by_tick.setdefault(t, {})
            tgt = r["target"]
            if tgt == "next_obs":
                d["ounc"] = r["uncertainty"]
                d["err_next"] = r["mean_abs_error"]
                d["upd"] = r["update_norm"]
            elif tgt == "action_consequence":
                d["runc"] = r["uncertainty"]
                d["err_rw"] = r["abs_error"]
            elif tgt == "retrieval_usefulness":
                d["uunc"] = r["uncertainty"]
                d["uhat"] = r["prediction"]
                d["benefit"] = r["actual"]
    ticks = []
    for key in sorted(by_tick):
        d = by_tick[key]
        assert set(d) == {"ounc", "err_next", "upd", "runc", "err_rw",
                          "uunc", "uhat", "benefit"}, (key, sorted(d))
        ticks.append(d)
    return ticks


def replay(ticks, learn=True, permute_errors=False):
    """Causal prequential replay. Returns per-domain lists of
    (p, o, u, err, ehat)."""
    est = CompetenceEstimator(EPS_OBS, EPS_RW, TAU_FAIL, learn=learn)
    n = len(ticks)
    if permute_errors:
        perm = [(i * 7919 + 13) % n for i in range(n)]
        assert sorted(perm) == list(range(n))
        err_pool = [ticks[perm[i]] for i in range(n)]
    else:
        err_pool = ticks
    rec = {d: [] for d in DOMAINS}
    upd_lag = 0.0
    for i, tk in enumerate(ticks):
        ek = err_pool[i]  # error-source tick (== tk unless permuted)
        ctx = {"ounc": tk["ounc"], "runc": tk["runc"],
               "uunc": tk["uunc"], "uhat": tk["uhat"]}
        pr = est.predict_tick(ctx)
        outs = {"next_obs": (ek["err_next"],
                             1.0 if ek["err_next"] <= EPS_OBS else 0.0),
                "action_consequence": (ek["err_rw"],
                                       1.0 if ek["err_rw"] <= EPS_RW else 0.0),
                "competence_failure": (ek["err_next"],
                                       1.0 if ek["err_next"] > TAU_FAIL else 0.0),
                "retrieval_usefulness": (abs(ek["benefit"]),
                                         1.0 if ek["benefit"] > 0 else 0.0)}
        for d in DOMAINS:
            err, o = outs[d]
            p = _clamp01(pr[d]["p"])
            u = p if d in U_IS_P else 1.0 - p
            rec[d].append((p, o, u, err, pr[d]["ehat"]))
        est.observe_tick(
            ctx,
            {"err_next": ek["err_next"], "err_rw": ek["err_rw"],
             "benefit": ek["benefit"]},
            upd_lag)
        upd_lag = tk["upd"]
    return rec


def score_domain(name, triples):
    n = len(triples)
    diag = {"n": n}
    if n == 0:
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": "zero triples collected"})
        return diag
    p = [t[0] for t in triples]
    o = [t[1] for t in triples]
    u = [t[2] for t in triples]
    err = [t[3] for t in triples]
    ehat = [t[4] for t in triples]
    if any(not math.isfinite(v) for v in p + o + u + err + ehat):
        diag.update({"verdict": "UNMEASURABLE",
                     "reason": "non-finite values"})
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
    b_e = brier(p, o)
    b_c = climatology_brier(o)
    b_half = brier([0.5] * n, o)
    bins, ece, mce = reliability(o, p)
    s = sum(pi - oi for pi, oi in zip(p, o)) / n
    corr_u = pearson(u, err)
    corr_e = pearson(ehat, err)
    if b_e >= b_c:
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
        "base_rate": base_rate, "brier_est": b_e, "brier_climo": b_c,
        "brier_const05": b_half, "skill_vs_climo": 1.0 - b_e / b_c,
        "ece": ece, "mce": mce, "signed_error": s,
        "corr_uncertainty_err": corr_u, "corr_ehat_err": corr_e,
        "verdict": verdict,
    })
    return diag


def run_seed(seed, out_dir):
    logp = os.path.join(out_dir, f"EXP-FP-0070.seed{seed}.predictions.jsonl")
    agent = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                       log_path=logp, affect="none")
    eps = run_closed_loop("changing_rule", agent, N_EPISODES, seed)
    if agent.plog:
        agent.plog.close()
    return logp, eps


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EXP-FP-0070 estimator battery")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--logs", nargs="*", default=None)
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

    pooled = {d: [] for d in DOMAINS}
    per_seed_n = {}
    for logp in logs:
        ticks = collect_ticks(logp)
        per_seed_n[logp] = len(ticks)
        rec = replay(ticks, learn=True)
        for d in DOMAINS:
            pooled[d].extend(rec[d])

    # determinism self-check: replay twice, byte-identical
    ticks0 = collect_ticks(logs[0])
    r1 = replay(ticks0, learn=True)
    r2 = replay(ticks0, learn=True)
    assert r1 == r2, "replay non-deterministic!"
    print("determinism self-check PASS")

    domains = {}
    for d in DOMAINS:
        sc = score_domain(d, pooled[d])
        domains[d] = sc
        print(f"{d}: n={sc['n']} verdict={sc.get('verdict')} "
              f"brier_est={sc.get('brier_est')} "
              f"brier_climo={sc.get('brier_climo')} "
              f"corr_u={sc.get('corr_uncertainty_err')} "
              f"corr_ehat={sc.get('corr_ehat_err')}")

    # kill arm 1: anchor-only must reproduce climatology to 1e-9
    anchor_ok = True
    for logp in logs:
        ticks = collect_ticks(logp)
        rec_a = replay(ticks, learn=False)
        for d in DOMAINS:
            p = [t[0] for t in rec_a[d]]
            o = [t[1] for t in rec_a[d]]
            if abs(brier(p, o) - climatology_brier(o)) > 1e-9:
                anchor_ok = False
                print(f"ANCHOR KILL FAILED on {d} {logp}")
    print(f"anchor-only kill arm: {'PASS' if anchor_ok else 'FAIL'}")

    # kill arm 2: error-permutation decomposition per domain
    perm_diag = {}
    for d in DOMAINS:
        b_intact = domains[d].get("brier_est")
        b_climo = domains[d].get("brier_climo")
        b_perm = None
        if b_intact is not None:
            pp = []
            for logp in logs:
                pp.extend(replay(collect_ticks(logp), learn=True,
                                 permute_errors=True)[d])
            b_perm = brier([t[0] for t in pp], [t[1] for t in pp])
        perm_diag[d] = {"brier_intact": b_intact, "brier_permuted": b_perm,
                        "brier_climo": b_climo,
                        "ordering_intact_lt_perm_lt_climo":
                            (b_perm is not None and b_intact < b_perm < b_climo)}
        print(f"perm {d}: intact={b_intact} permuted={b_perm} climo={b_climo} "
              f"-> {perm_diag[d]['ordering_intact_lt_perm_lt_climo']}")

    cfg = {"experiment_id": "EXP-FP-0070",
           "contract_version": "1.0.0",
           "env": "changing_rule", "env_version": "1.0.0",
           "agent": "arch_b", "affect": "none",
           "episodes_per_seed": N_EPISODES, "primary_seeds": SEEDS,
           "thresholds": {"EPS_OBS": EPS_OBS, "EPS_RW": EPS_RW,
                          "TAU_FAIL": TAU_FAIL},
           "bins": N_BINS,
           "estimator": "competence_estimator.CompetenceEstimator variant A "
                        "(online logistic + log-linear, climatology-anchored, "
                        "normalized SGD eta_p=0.1/eta_e=0.05)"}
    results = {
        "experiment_id": "EXP-FP-0070",
        "started_utc": started,
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "seeds": SEEDS,
        "logs": logs,
        "per_seed_ticks": per_seed_n,
        "domains": domains,
        "kill_arms": {"anchor_reproduces_climatology_1e9": anchor_ok,
                      "error_permutation_decomposition": perm_diag},
        "determinism_self_check": True,
    }
    outp = args.out or os.path.join(HERE, "competence_estimator_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"results -> {outp}")

    # preregistered win gates
    beats = sum(1 for d in DOMAINS
                if domains[d].get("brier_est", 1e9)
                < domains[d].get("brier_climo", -1e9))
    coupled = sum(1 for d in DOMAINS
                  if (domains[d].get("corr_uncertainty_err") or 0) >= 0.40)
    print(f"WIN GATES: brier-beats={beats}/4 (need >=3), "
          f"coupled>={coupled}/4 (need >=3)")
    return results, cfg, started


if __name__ == "__main__":
    main()
