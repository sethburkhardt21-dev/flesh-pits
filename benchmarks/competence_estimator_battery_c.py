"""Competence-estimator battery — EXP-FP-0072 (variant C).

Variant C: channel-decomposed REST error features. D1/D3's error-channel
features (EMA_fast/slow, z-surprise) and ehat head track rest_err =
mean(|signed_error[2:6]|) (obs dims 2-5, excluding the pure-RNG cue dims
0-1); the p heads still train on the preregistered total-error events
(domains/events UNCHANGED -- no goalpost moving). rest_err history is
maintained causally from past ticks' recorded signed_errors; tick t's own
signed_error is used only in post-prediction training (unit-tested).

Reports additionally corr(ehat, rest_err) for D1/D3 (diagnostic: honest
coupling on the predictable component).

Deterministic. Stdlib only.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB_DIR = os.path.join(HERE, "..", "prototypes", "architecture-b")
FP_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(AB_DIR))
sys.path.insert(0, os.path.join(os.path.abspath(FP_DIR), "experiments"))
sys.path.insert(0, HERE)

import competence_estimator_battery_b as B  # noqa: E402
from competence_estimator_battery_b import (  # noqa: E402
    EPS_OBS, EPS_RW, TAU_FAIL, N_EPISODES, N_BINS, DOMAINS, U_IS_P, OUT_DIR,
    brier, climatology_brier, score_domain, _clamp01, collect_merged,
    run_seed, equivalence_check,
)
from env_interface import config_hash  # noqa: E402
from competence_estimator import CompetenceEstimator  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

SEEDS = [73211, 73212, 73213, 73214]
TAG = "EXP-FP-0072"


def rest_of(tk):
    se = tk["signed_error"]
    return sum(abs(x) for x in se[2:6]) / 4.0


def collect_merged_c(log_path, sidecar_path):
    """collect_merged + per-tick recorded signed_error (outcome; the replay
    uses it only in post-prediction training)."""
    ticks = collect_merged(log_path, sidecar_path)
    by_tick = {}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["target"] == "next_obs":
                by_tick[(r["tick"], r["episode"])] = r["signed_error"]
    return ticks, by_tick


def replay_c(ticks, by_tick, learn=True, permute_errors=False,
             novelty_zeroed=False):
    est = CompetenceEstimator(EPS_OBS, EPS_RW, TAU_FAIL, learn=learn,
                              feature_mode="C")
    n = len(ticks)
    order = sorted(by_tick)
    assert len(order) == n
    se_list = [by_tick[k] for k in order]
    if permute_errors:
        perm = [(i * 7919 + 13) % n for i in range(n)]
        assert sorted(perm) == list(range(n))
        err_pool = [ticks[perm[i]] for i in range(n)]
        se_pool = [se_list[perm[i]] for i in range(n)]
    else:
        err_pool = ticks
        se_pool = se_list
    rec = {d: [] for d in DOMAINS}
    upd_lag = 0.0
    for i, tk in enumerate(ticks):
        ek = err_pool[i]
        sek = se_pool[i]
        rest = sum(abs(x) for x in sek[2:6]) / 4.0
        ctx = {"ounc": tk["ounc"], "runc": tk["runc"],
               "uunc": tk["uunc"], "uhat": tk["uhat"],
               "signed_error": sek}
        if not novelty_zeroed:
            ctx.update({"mean_similarity": tk["mean_similarity"],
                        "n_nbrs": tk["n_nbrs"],
                        "retrieval_used": tk["retrieval_used"],
                        "e1_ema": tk["e1_ema"]})
        pr = est.predict_tick(ctx)
        outs = {"next_obs": (ek["err_next"],
                             1.0 if ek["err_next"] <= EPS_OBS else 0.0,
                             rest),
                "action_consequence": (ek["err_rw"],
                                       1.0 if ek["err_rw"] <= EPS_RW else 0.0,
                                       None),
                "competence_failure": (ek["err_next"],
                                       1.0 if ek["err_next"] > TAU_FAIL else 0.0,
                                       rest),
                "retrieval_usefulness": (abs(ek["benefit"]),
                                         1.0 if ek["benefit"] > 0 else 0.0,
                                         None)}
        for d in DOMAINS:
            err, o, rs = outs[d]
            p = _clamp01(pr[d]["p"])
            u = p if d in U_IS_P else 1.0 - p
            rec[d].append((p, o, u, err, pr[d]["ehat"], rs))
        est.observe_tick(
            ctx,
            {"err_next": ek["err_next"], "err_rw": ek["err_rw"],
             "benefit": ek["benefit"]},
            upd_lag)
        upd_lag = tk["upd"]
    return rec


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = (sum((x - mx) ** 2 for x in xs)
           * sum((y - my) ** 2 for y in ys)) ** 0.5
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EXP-FP-0072 estimator battery")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    logs = []
    episodes = {}
    if args.run:
        for seed in SEEDS:
            print(f"--- seed {seed} ---", flush=True)
            logp, scp, eps = run_seed(seed, OUT_DIR, tag=TAG)
            logs.append((logp, scp))
            episodes[str(seed)] = eps
    else:
        raise SystemExit("pass --run (variant C needs signed_error replays)")

    pooled = {d: [] for d in DOMAINS}
    merged = []
    for logp, scp in logs:
        ticks, by_tick = collect_merged_c(logp, scp)
        merged.append((ticks, by_tick))
        rec = replay_c(ticks, by_tick, learn=True)
        for d in DOMAINS:
            pooled[d].extend(rec[d])

    ticks0, bt0 = merged[0]
    r1 = replay_c(ticks0, bt0, learn=True)
    r2 = replay_c(ticks0, bt0, learn=True)
    assert r1 == r2, "replay non-deterministic!"
    print("determinism self-check PASS")

    domains = {}
    for d in DOMAINS:
        triples = [(t[0], t[1], t[2], t[3], t[4]) for t in pooled[d]]
        sc = score_domain(d, triples)
        # diagnostic: corr(ehat, rest_err) for D1/D3
        rests = [t[5] for t in pooled[d]]
        ehat = [t[4] for t in pooled[d]]
        sc["corr_ehat_rest"] = (pearson(ehat, rests)
                                if rests[0] is not None else None)
        domains[d] = sc
        print(f"{d}: n={sc['n']} verdict={sc.get('verdict')} "
              f"brier_est={sc.get('brier_est')} "
              f"brier_climo={sc.get('brier_climo')} "
              f"corr_u={sc.get('corr_uncertainty_err')} "
              f"corr_ehat={sc.get('corr_ehat_err')} "
              f"corr_ehat_rest={sc.get('corr_ehat_rest')}")

    anchor_ok = True
    for ticks, by_tick in merged:
        rec_a = replay_c(ticks, by_tick, learn=False)
        for d in DOMAINS:
            p = [t[0] for t in rec_a[d]]
            o = [t[1] for t in rec_a[d]]
            if abs(brier(p, o) - climatology_brier(o)) > 1e-9:
                anchor_ok = False
                print(f"ANCHOR KILL FAILED on {d}")
    print(f"anchor-only kill arm: {'PASS' if anchor_ok else 'FAIL'}")

    perm_diag, nov_diag = {}, {}
    for d in DOMAINS:
        b_intact = domains[d].get("brier_est")
        b_climo = domains[d].get("brier_climo")
        pp, nn = [], []
        for ticks, by_tick in merged:
            pp.extend(replay_c(ticks, by_tick, learn=True,
                               permute_errors=True)[d])
            nn.extend(replay_c(ticks, by_tick, learn=True,
                               novelty_zeroed=True)[d])
        b_perm = brier([t[0] for t in pp], [t[1] for t in pp]) if pp else None
        b_nov0 = brier([t[0] for t in nn], [t[1] for t in nn]) if nn else None
        perm_diag[d] = {"brier_intact": b_intact, "brier_permuted": b_perm,
                        "brier_climo": b_climo,
                        "ordering_intact_lt_perm_lt_climo":
                            (b_perm is not None and b_intact < b_perm < b_climo)}
        nov_diag[d] = {"brier_novelty_zeroed": b_nov0,
                       "novelty_load_bearing": (b_nov0 is not None
                                                and b_intact is not None
                                                and b_nov0 > b_intact + 1e-6)}
        print(f"perm {d}: intact={b_intact} permuted={b_perm} climo={b_climo}")
        print(f"nov0 {d}: novelty_zeroed={b_nov0} -> "
              f"load_bearing={nov_diag[d]['novelty_load_bearing']}")

    cfg = {"experiment_id": "EXP-FP-0072",
           "contract_version": "1.0.0",
           "env": "changing_rule", "env_version": "1.0.0",
           "agent": "arch_b_instrumented (equivalence-proven in 0071)",
           "affect": "none",
           "episodes_per_seed": N_EPISODES, "primary_seeds": SEEDS,
           "thresholds": {"EPS_OBS": EPS_OBS, "EPS_RW": EPS_RW,
                          "TAU_FAIL": TAU_FAIL},
           "bins": N_BINS,
           "estimator": "competence_estimator.CompetenceEstimator variant C "
                        "feature_mode='C' (REST-channel features + ehat on "
                        "rest for D1/D3), normalized SGD eta_p=0.1/eta_e=0.05"}
    results = {
        "experiment_id": "EXP-FP-0072",
        "started_utc": started,
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "seeds": SEEDS,
        "logs": [lp for lp, _ in logs],
        "sidecars": [sc for _, sc in logs],
        "domains": domains,
        "kill_arms": {"anchor_reproduces_climatology_1e9": anchor_ok,
                      "error_permutation_decomposition": perm_diag,
                      "novelty_zeroed_ablation": nov_diag},
        "determinism_self_check": True,
    }
    outp = args.out or os.path.join(HERE, "competence_estimator_c_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"results -> {outp}")

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
