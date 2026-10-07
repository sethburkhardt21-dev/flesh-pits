"""Competence-estimator battery — EXP-FP-0100 (family 1: estimand re-scoping).

Family 1 of the post-0072 re-scoping program. The 0070-0072 arc closed
ALL-NEGATIVE: the corr>=0.40 coupling target is UNREACHABLE on changing_rule's
total error because pure-RNG cue channels (obs dims 0-1) contribute 60.5% of
|err| variance -- a SIGNAL CEILING, not a learning failure.

EXP-FP-0100 re-scopes the estimand: competence = PREDICTABLE error
(rest-channel dims 2-5), not total error. The estimator (feature_mode='D')
is identical to variant C EXCEPT the D1/D3 p heads train on the re-scoped
REST-error events. D2 (reward) and D4 (benefit) are unchanged (no cue
channels in their estimands). D1/D3 thresholds are frozen from the
EXP-FP-CALIB-01 history (median/p90 of mean_abs_rest_err; derivation
faithfulness verified in the preregistration).

Preregistered: experiments/preregistration_EXP-FP-0100.json (sealed pre-run).
Win gates: Brier beats sequential climatology on >=3/4 domains AND
corr(predicted_uncertainty, predictable|err|) >= 0.40 on BOTH re-scoped
domains (D1, D3).

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

import competence_estimator_battery_c as C  # noqa: E402
from competence_estimator_battery_c import (  # noqa: E402
    collect_merged_c, rest_of, replay_c, pearson,
)
from competence_estimator_battery_b import (  # noqa: E402
    EPS_RW, N_EPISODES, DOMAINS, U_IS_P, brier, climatology_brier,
    score_domain, _clamp01, make_instrumented, OUT_DIR,
)
from env_interface import config_hash  # noqa: E402
from experiments import ENVS_BY_NAME, run_closed_loop  # noqa: E402
from competence_estimator import CompetenceEstimator  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

SEEDS = [73301, 73302, 73303, 73304]
TAG = "EXP-FP-0100"
# Frozen from the EXP-FP-CALIB-01 history (see preregistration).
EPS_OBS_REST = 0.1387   # hist median of mean_abs_rest_err, 1950 ticks
TAU_FAIL_REST = 0.2896  # hist p90 of mean_abs_rest_err, same ticks


def run_seed_d(seed, out_dir):
    logp = os.path.join(out_dir, f"EXP-FP-0100.seed{seed}.predictions.jsonl")
    scp = os.path.join(out_dir, f"EXP-FP-0100.seed{seed}.sidecar.jsonl")
    agent = make_instrumented(ENVS_BY_NAME["changing_rule"], seed=seed,
                              log_path=logp, sidecar_path=scp, affect="none")
    eps = run_closed_loop("changing_rule", agent, N_EPISODES, seed)
    if agent.plog:
        agent.plog.close()
    agent.close_sidecar()
    return logp, scp, eps


def replay_d(ticks, by_tick, learn=True, permute_errors=False,
             novelty_zeroed=False):
    """Causal prequential replay, feature_mode='D'. D1/D3's p heads train on
    the re-scoped rest events; coupling is against predictable|err|."""
    est = CompetenceEstimator(EPS_OBS_REST, EPS_RW, TAU_FAIL_REST,
                              learn=learn, feature_mode="D")
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
        rest = rest_of({"signed_error": sek})
        ctx = {"ounc": tk["ounc"], "runc": tk["runc"],
               "uunc": tk["uunc"], "uhat": tk["uhat"],
               "signed_error": sek}
        if not novelty_zeroed:
            ctx.update({"mean_similarity": tk["mean_similarity"],
                        "n_nbrs": tk["n_nbrs"],
                        "retrieval_used": tk["retrieval_used"],
                        "e1_ema": tk["e1_ema"]})
        pr = est.predict_tick(ctx)
        outs = {"next_obs": (rest, 1.0 if rest <= EPS_OBS_REST else 0.0),
                "action_consequence": (ek["err_rw"],
                                       1.0 if ek["err_rw"] <= EPS_RW else 0.0),
                "competence_failure": (rest,
                                       1.0 if rest > TAU_FAIL_REST else 0.0),
                "retrieval_usefulness": (abs(ek["benefit"]),
                                         1.0 if ek["benefit"] > 0 else 0.0)}
        for d in DOMAINS:
            pred_err, o = outs[d]
            p = _clamp01(pr[d]["p"])
            u = p if d in U_IS_P else 1.0 - p
            rec[d].append((p, o, u, pred_err, pr[d]["ehat"]))
        est.observe_tick(
            ctx,
            {"err_next": ek["err_next"], "err_rw": ek["err_rw"],
             "benefit": ek["benefit"]},
            upd_lag)
        upd_lag = tk["upd"]
    return rec


def reference_mode_c_on_rescoped(logs):
    """EXP-FP-0072 variant-C estimator replayed on the 0100 logs, scored on
    the RE-SCOPED estimand (diagnostic, not gated). Mode C's p heads predict
    total-error events; we score its Brier vs the rest events and its
    coupling vs rest_err for D1/D3."""
    ref = {}
    for d in DOMAINS:
        allp, allo, allu, allerr, allehat = [], [], [], [], []
        for logp, scp in logs:
            ticks, by_tick = collect_merged_c(logp, scp)
            rec = replay_c(ticks, by_tick, learn=True)[d]
            for (p, o, u, err, ehat, rs) in rec:
                if d in ("next_obs", "competence_failure"):
                    o_r = (1.0 if rs <= EPS_OBS_REST else 0.0) if d == "next_obs" \
                        else (1.0 if rs > TAU_FAIL_REST else 0.0)
                    allp.append(p)
                    allo.append(o_r)
                    allu.append(u)
                    allerr.append(rs)
                    allehat.append(ehat)
                else:
                    allp.append(p)
                    allo.append(o)
                    allu.append(u)
                    allerr.append(err)
                    allehat.append(ehat)
        triples = list(zip(allp, allo, allu, allerr, allehat))
        sc = score_domain(d, triples)
        ref[d] = sc
    return ref


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EXP-FP-0100 estimator battery")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--receipts", default=None,
                    help="receipts dir (default: <flesh-pits>/receipts)")
    args = ap.parse_args(argv)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    logs = []
    episodes = {}
    if args.run:
        for seed in SEEDS:
            print(f"--- seed {seed} ---", flush=True)
            logp, scp, eps = run_seed_d(seed, OUT_DIR)
            logs.append((logp, scp))
            episodes[str(seed)] = eps
    else:
        raise SystemExit("pass --run")

    pooled = {d: [] for d in DOMAINS}
    merged = []
    for logp, scp in logs:
        ticks, by_tick = collect_merged_c(logp, scp)
        merged.append((ticks, by_tick))
        rec = replay_d(ticks, by_tick, learn=True)
        for d in DOMAINS:
            pooled[d].extend(rec[d])

    ticks0, bt0 = merged[0]
    r1 = replay_d(ticks0, bt0, learn=True)
    r2 = replay_d(ticks0, bt0, learn=True)
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

    # kill arm 1: anchor-only (learn=False, mode D) reproduces climatology
    # on the re-scoped events to 1e-9
    anchor_ok = True
    for ticks, by_tick in merged:
        rec_a = replay_d(ticks, by_tick, learn=False)
        for d in DOMAINS:
            p = [t[0] for t in rec_a[d]]
            o = [t[1] for t in rec_a[d]]
            if abs(brier(p, o) - climatology_brier(o)) > 1e-9:
                anchor_ok = False
                print(f"ANCHOR KILL FAILED on {d}")
    print(f"anchor-only kill arm: {'PASS' if anchor_ok else 'FAIL'}")

    # kill arm 2: error-permutation decomposition; novelty-zeroed ablation
    perm_diag, nov_diag = {}, {}
    for d in DOMAINS:
        b_intact = domains[d].get("brier_est")
        b_climo = domains[d].get("brier_climo")
        pp, nn = [], []
        for ticks, by_tick in merged:
            pp.extend(replay_d(ticks, by_tick, learn=True,
                               permute_errors=True)[d])
            nn.extend(replay_d(ticks, by_tick, learn=True,
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

    # reference arm: 0072 variant-C estimator on the same logs, re-scoped
    ref = reference_mode_c_on_rescoped(logs)
    for d in DOMAINS:
        print(f"ref-modeC {d}: brier={ref[d].get('brier_est')} "
              f"climo={ref[d].get('brier_climo')} "
              f"corr_u={ref[d].get('corr_uncertainty_err')}")

    beats = sum(1 for d in DOMAINS
                if domains[d].get("brier_est", 1e9)
                < domains[d].get("brier_climo", -1e9))
    coupled_d1 = (domains["next_obs"].get("corr_uncertainty_err") or 0) >= 0.40
    coupled_d3 = (domains["competence_failure"].get("corr_uncertainty_err") or 0) >= 0.40
    win = beats >= 3 and coupled_d1 and coupled_d3
    partial = beats >= 3 and not (coupled_d1 and coupled_d3)
    verdict = ("SUCCESS" if win else "PARTIAL" if partial else "NULL_HOLDS")
    print(f"WIN GATES: brier-beats={beats}/4 (need >=3), "
          f"D1-coupling={coupled_d1}, D3-coupling={coupled_d3} -> {verdict}")

    cfg = {"experiment_id": "EXP-FP-0100",
           "contract_version": "1.0.0",
           "env": "changing_rule", "env_version": "1.0.0",
           "agent": "arch_b_instrumented (equivalence-proven in 0071)",
           "affect": "none",
           "episodes_per_seed": N_EPISODES, "primary_seeds": SEEDS,
           "thresholds": {"EPS_OBS_REST": EPS_OBS_REST,
                          "TAU_FAIL_REST": TAU_FAIL_REST, "EPS_RW": EPS_RW},
           "bins": 10,
           "estimator": "competence_estimator.CompetenceEstimator feature_mode='D' "
                        "(mode-C feature set; D1/D3 p heads train on re-scoped "
                        "rest-error events), normalized SGD eta_p=0.1/eta_e=0.05"}
    results = {
        "experiment_id": "EXP-FP-0100",
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
        "reference_modeC_on_rescoped": ref,
        "determinism_self_check": True,
        "win_gates": {"brier_beats_ge3": beats >= 3,
                      "d1_coupling_ge040": coupled_d1,
                      "d3_coupling_ge040": coupled_d3,
                      "verdict": verdict},
    }
    outp = args.out or os.path.join(HERE, "competence_estimator_d_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"results -> {outp}")

    # hash-chained receipt (claim IDs via id_registry; EXP-FP-0100 claimed)
    receipt_result = {
        "experiment_id": "EXP-FP-0100",
        "config": cfg,
        "config_hash": config_hash(cfg),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [{"seed": s, "prediction_log": lp, "sidecar": sc,
                      "n_episodes": episodes.get(str(s))}
                     for (lp, sc), s in zip(logs, SEEDS)],
        "summary": {"domains": domains,
                    "kill_arms": results["kill_arms"],
                    "win_gates": results["win_gates"]},
    }
    receipts = args.receipts or os.path.join(FP_DIR, "receipts")
    hypothesis = ("Estimand re-scoping (exclude the pure-RNG cue channels from "
                  "the estimand; competence = predictable error) couples the "
                  "causal online competence estimator to error magnitude: Brier "
                  "beats sequential climatology on >=3/4 domains AND "
                  "corr(predicted_uncertainty, predictable|err|) >= 0.40 on the "
                  "re-scoped D1/D3 estimand.")
    null = ("Re-scoping changes nothing (the estimator is theater on the "
            "re-scoped estimand too).")
    metric = ("Per-domain Brier_est vs sequential-climatology Brier_climo on the "
              "re-scoped events (D1/D3); corr(predicted_uncertainty, "
              "predictable|err|) gated at 0.40 on D1/D3; ECE/MCE reported. "
              "Preregistered: experiments/preregistration_EXP-FP-0100.json "
              "(sealed pre-run).")
    path = write_receipt(receipt_result, receipts, hypothesis=hypothesis,
                         null=null, preregistered_metric=metric,
                         baseline="sequential climatology (Laplace running "
                                  "base rate, per domain per seed) on the "
                                  "re-scoped events",
                         conditions=("changing_rule v1.0.0, "
                                     "InstrumentedArchB (equivalence-proven), "
                                     "affect=none, 10 eps x 4 fresh seeds "
                                     "{73301,73302,73303,73304}. "
                                     f"{verdict}."))
    print(f"receipt: {path}")
    ok, problems = verify_chain(receipts)
    print(f"chain: {'OK' if ok else problems}")
    return results, cfg, started


if __name__ == "__main__":
    main()
