"""Competence-estimator battery — EXP-FP-0071 (variant B).

Variant B adds the agent's prediction-time novelty/state context
(mean_similarity, n_nbrs, retrieval_used, e1_ema) via the additive
InstrumentedArchB sidecar (agent.py untouched). Pre-run equivalence check
proves the instrumentation is behaviorally inert (PredictionLog JSONL
byte-identical between ArchB and InstrumentedArchB on a throwaway seed).

Otherwise identical protocol to EXP-FP-0070: causal prequential replay with
feature_mode='B', same binary events / frozen thresholds / climatology
baseline / ECE-MCE gates / corr>=0.40 coupling gate, same two kill arms,
plus a novelty-zeroed ablation (sidecar fields forced to 0).

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
sys.path.insert(0, HERE)

from competence_estimator_battery import (  # noqa: E402
    EPS_OBS, EPS_RW, TAU_FAIL, N_EPISODES, N_BINS, DOMAINS, U_IS_P,
    brier, climatology_brier, score_domain, _clamp01,
)
from env_interface import config_hash, derive_seed  # noqa: E402
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from competence_estimator import CompetenceEstimator  # noqa: E402
from instrumented_arch_b import InstrumentedArchB  # noqa: E402
from agent import ArchB  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

SEEDS = [73205, 73206, 73208, 73209]
EQUIV_SEED = 73210  # throwaway seed for the behavior-equivalence check only
OUT_DIR = os.path.join(os.path.abspath(AB_DIR), "experiments_out")


def make_instrumented(env_cls, seed=0, log_path=None, sidecar_path=None,
                      **kwargs):
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    return InstrumentedArchB(observation_space=space, n_actions=n_actions,
                             env_name=env_cls.NAME, seed=seed,
                             log_path=log_path, sidecar_path=sidecar_path,
                             **kwargs)


def collect_merged(log_path, sidecar_path):
    """PredictionLog JSONL + sidecar -> ordered per-tick dicts with
    prediction-time novelty context."""
    side = {}
    with open(sidecar_path) as fh:
        for line in fh:
            r = json.loads(line)
            side[(r["tick"], r["episode"])] = r
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
        s = side.get(key)
        assert s is not None, f"sidecar missing tick {key}"
        d.update({"mean_similarity": s["mean_similarity"],
                  "n_nbrs": s["n_nbrs"],
                  "retrieval_used": s["retrieval_used"],
                  "e1_ema": s["e1_ema"]})
        ticks.append(d)
    # every sidecar tick must have a log tick
    assert set(side) == set(by_tick), "sidecar/log tick mismatch"
    return ticks


def replay_b(ticks, learn=True, permute_errors=False, novelty_zeroed=False):
    est = CompetenceEstimator(EPS_OBS, EPS_RW, TAU_FAIL, learn=learn,
                              feature_mode="B")
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
        ek = err_pool[i]
        ctx = {"ounc": tk["ounc"], "runc": tk["runc"],
               "uunc": tk["uunc"], "uhat": tk["uhat"]}
        if not novelty_zeroed:
            ctx.update({"mean_similarity": tk["mean_similarity"],
                        "n_nbrs": tk["n_nbrs"],
                        "retrieval_used": tk["retrieval_used"],
                        "e1_ema": tk["e1_ema"]})
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


def run_seed(seed, out_dir, tag):
    logp = os.path.join(out_dir, f"EXP-FP-0071{tag}.seed{seed}.predictions.jsonl")
    scp = os.path.join(out_dir, f"EXP-FP-0071{tag}.seed{seed}.sidecar.jsonl")
    agent = make_instrumented(ENVS_BY_NAME["changing_rule"], seed=seed,
                              log_path=logp, sidecar_path=scp, affect="none")
    eps = run_closed_loop("changing_rule", agent, N_EPISODES, seed)
    if agent.plog:
        agent.plog.close()
    agent.close_sidecar()
    return logp, scp, eps


def equivalence_check(out_dir):
    """ArchB vs InstrumentedArchB on a throwaway seed: PredictionLog JSONL
    must be byte-identical (instrumentation is behaviorally inert)."""
    lp_a = os.path.join(out_dir, "EXP-FP-0071.equivA.predictions.jsonl")
    lp_b = os.path.join(out_dir, "EXP-FP-0071.equivB.predictions.jsonl")
    sc_b = os.path.join(out_dir, "EXP-FP-0071.equivB.sidecar.jsonl")
    a = make_agent(ENVS_BY_NAME["changing_rule"], seed=EQUIV_SEED,
                   log_path=lp_a, affect="none")
    run_closed_loop("changing_rule", a, N_EPISODES, EQUIV_SEED)
    a.plog.close()
    b = make_instrumented(ENVS_BY_NAME["changing_rule"], seed=EQUIV_SEED,
                          log_path=lp_b, sidecar_path=sc_b, affect="none")
    run_closed_loop("changing_rule", b, N_EPISODES, EQUIV_SEED)
    b.plog.close()
    b.close_sidecar()
    da = open(lp_a, "rb").read()
    db = open(lp_b, "rb").read()
    ok = da == db
    print(f"equivalence check (seed {EQUIV_SEED}): "
          f"{'PASS byte-identical' if ok else 'FAIL DIVERGED'} "
          f"({len(da)} bytes)")
    return ok


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EXP-FP-0071 estimator battery")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="prediction JSONL logs; sidecars inferred by path")
    args = ap.parse_args(argv)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)

    equiv_ok = equivalence_check(OUT_DIR)
    assert equiv_ok, "instrumentation is NOT behaviorally inert -- STOP"

    logs = []
    episodes = {}
    if args.logs:
        for lp in args.logs:
            assert ".seed" in lp
            sc = lp.replace(".predictions.jsonl", ".sidecar.jsonl")
            logs.append((lp, sc))
    elif args.run:
        for seed in SEEDS:
            print(f"--- seed {seed} ---", flush=True)
            logp, scp, eps = run_seed(seed, OUT_DIR, tag="")
            logs.append((logp, scp))
            episodes[str(seed)] = eps
    else:
        raise SystemExit("pass --run or --logs")

    pooled = {d: [] for d in DOMAINS}
    for logp, scp in logs:
        ticks = collect_merged(logp, scp)
        rec = replay_b(ticks, learn=True)
        for d in DOMAINS:
            pooled[d].extend(rec[d])

    ticks0 = collect_merged(*logs[0])
    r1 = replay_b(ticks0, learn=True)
    r2 = replay_b(ticks0, learn=True)
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

    anchor_ok = True
    for logp, scp in logs:
        rec_a = replay_b(collect_merged(logp, scp), learn=False)
        for d in DOMAINS:
            p = [t[0] for t in rec_a[d]]
            o = [t[1] for t in rec_a[d]]
            if abs(brier(p, o) - climatology_brier(o)) > 1e-9:
                anchor_ok = False
                print(f"ANCHOR KILL FAILED on {d} {logp}")
    print(f"anchor-only kill arm: {'PASS' if anchor_ok else 'FAIL'}")

    perm_diag = {}
    nov_diag = {}
    for d in DOMAINS:
        b_intact = domains[d].get("brier_est")
        b_climo = domains[d].get("brier_climo")
        pp, nn = [], []
        for logp, scp in logs:
            tk = collect_merged(logp, scp)
            pp.extend(replay_b(tk, learn=True, permute_errors=True)[d])
            nn.extend(replay_b(tk, learn=True, novelty_zeroed=True)[d])
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

    cfg = {"experiment_id": "EXP-FP-0071",
           "contract_version": "1.0.0",
           "env": "changing_rule", "env_version": "1.0.0",
           "agent": "arch_b_instrumented (additive subclass; equiv-checked)",
           "affect": "none",
           "episodes_per_seed": N_EPISODES, "primary_seeds": SEEDS,
           "thresholds": {"EPS_OBS": EPS_OBS, "EPS_RW": EPS_RW,
                          "TAU_FAIL": TAU_FAIL},
           "bins": N_BINS,
           "estimator": "competence_estimator.CompetenceEstimator variant B "
                        "feature_mode='B' (+novelty features), normalized SGD "
                        "eta_p=0.1/eta_e=0.05",
           "equivalence_check": {"seed": EQUIV_SEED, "byte_identical": True}}
    results = {
        "experiment_id": "EXP-FP-0071",
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
        "equivalence_check_pass": equiv_ok,
    }
    outp = args.out or os.path.join(HERE, "competence_estimator_b_results.json")
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
