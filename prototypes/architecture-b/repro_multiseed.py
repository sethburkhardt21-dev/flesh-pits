"""REPRODUCTION driver — Architecture B K1,K2,K3,K5,C1,C2,C4,BAR on 5 fresh seeds.

Reuses the ORIGINAL helper functions from experiments.py (run_closed_loop,
run_replay, collect_transitions, _white_noise_transitions, _train_eval,
make_agent, _mean) — the experiment logic below is the original code with
the hardcoded primary/agent seeds replaced by a `seed` parameter. All RNG
streams (env episodes via derive_seed, agent init, agent reset, white-noise
data, shuffle RNG, train/eval agent seeds) derive from the fresh primary
seed, documented per experiment. Fresh seeds are distinct from the
originals (101-109).

A verdict REPRODUCES if the original verdict holds on >=4/5 seeds.
Receipts: receipts/repro_EXP-AB-*.json (original schema + reproduction).
"""
import argparse
import copy
import datetime
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import experiments as E
from experiments import (ENVS_BY_NAME, make_agent, run_closed_loop,
                         run_replay, collect_transitions,
                         _white_noise_transitions, _train_eval, _mean,
                         new_rng, ArchB, derive_seed, OUT_DIR, RECEIPTS_DIR,
                         config_hash, CONTRACT_VERSION)

SEEDS = {
    "K1": [72001, 72002, 72003, 72004, 72005],
    "K2": [72101, 72102, 72103, 72104, 72105],
    "K3": [72201, 72202, 72203, 72204, 72205],
    "K5": [72301, 72302, 72303, 72304, 72305],
    "C1": [72401, 72402, 72403, 72404, 72405],
    "C2": [72501, 72502, 72503, 72504, 72505],
    "C4": [72601, 72602, 72603, 72604, 72605],
    "BAR": [72701, 72702, 72703, 72704, 72705],
}

ORIG_SEED = {"K1": 101, "K2": 102, "K3": 103, "K5": 105, "C1": 106,
             "C2": 107, "C4": 108, "BAR": 109}


def write_repro_receipt(exp_id, prereg, per_seed, aggregate, interpretation,
                        limitations):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {
        "experiment_id": exp_id,
        "architecture": "B",
        "contract_version": CONTRACT_VERSION,
        "reproduction": True,
        "original_primary_seed": ORIG_SEED[exp_id.split("-")[-1]],
        "preregistered": prereg,
        "config_hash": config_hash(prereg.get("conditions", {})),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "interpretation": interpretation,
        "limitations": limitations,
        "written_utc": now,
    }
    path = os.path.join(RECEIPTS_DIR, f"repro_{exp_id}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


def prereg_copy(name):
    return copy.deepcopy(getattr(E, f"{name}_PREREG"))


# ---------------------------------------------------------------- K1
def k1_one(seed):
    learn = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed, affect="none")
    eps_learn = run_closed_loop("changing_rule", learn, 10, seed)
    frozen = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed, frozen=True,
                        affect="none")
    eps_frozen = run_closed_loop("changing_rule", frozen, 10, seed)
    post_learn = _mean([e["return"] for e in eps_learn[5:10]])
    post_frozen = _mean([e["return"] for e in eps_frozen[5:10]])
    kill = not (post_learn > post_frozen + 1e-9)
    return {"seed": seed,
            "mean_return_postflip_learn": post_learn,
            "mean_return_postflip_frozen": post_frozen,
            "delta": post_learn - post_frozen,
            "kill": bool(kill), "verdict_holds": not kill,
            "episodes_learn": [e["return"] for e in eps_learn],
            "episodes_frozen": [e["return"] for e in eps_frozen]}


# ---------------------------------------------------------------- K2
def k2_one(seed):
    learn = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed, affect="none")
    eps = run_closed_loop("changing_rule", learn, 10, seed, record_actions=True)
    err_learn = list(learn.agg["e0_abs"])
    frozen = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed, frozen=True,
                        affect="none")
    run_replay("changing_rule", frozen, 10, seed,
               [ep["actions"] for ep in eps])
    err_frozen = list(frozen.agg["e0_abs"])
    m_learn, m_frozen = _mean(err_learn), _mean(err_frozen)
    kill = not (m_learn < m_frozen - 1e-9)
    return {"seed": seed, "mean_abs_e0_learn": m_learn,
            "mean_abs_e0_frozen": m_frozen, "delta": m_frozen - m_learn,
            "n_ticks": len(err_learn), "kill": bool(kill),
            "verdict_holds": not kill}


# ---------------------------------------------------------------- K3
def k3_one(seed):
    env_name = "changing_rule"
    env_cls = ENVS_BY_NAME[env_name]
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    obs_dim = sum(1 if s["type"] == "scalar" else
                  s["shape"] if s["type"] == "vector" else
                  len(s["values"]) for s in space.values())
    ref = make_agent(env_cls, seed=seed)
    all_t = collect_transitions(env_name, 12, seed, ref)
    train, heldout = all_t[:300], all_t[300:420]
    wn_train = _white_noise_transitions(300, obs_dim, n_actions, seed + 3)
    wn_held = _white_noise_transitions(120, obs_dim, n_actions, seed + 4)
    s_intact = _train_eval(train, heldout, False, seed + 1, env_name,
                           obs_dim, n_actions)
    s_les = _train_eval(train, heldout, True, seed + 1, env_name,
                        obs_dim, n_actions)
    w_intact = _train_eval(wn_train, wn_held, False, seed + 2, env_name,
                           obs_dim, n_actions)
    w_les = _train_eval(wn_train, wn_held, True, seed + 2, env_name,
                        obs_dim, n_actions)
    kill = not (s_les > s_intact + 1e-9)
    return {"seed": seed, "structured_intact": s_intact,
            "structured_lesioned": s_les, "structured_delta": s_les - s_intact,
            "whitenoise_intact": w_intact, "whitenoise_lesioned": w_les,
            "whitenoise_delta": w_les - w_intact,
            "kill": bool(kill), "verdict_holds": not kill}


# ---------------------------------------------------------------- K5
def k5_one(seed):
    env_name = "changing_rule"
    env_cls = ENVS_BY_NAME[env_name]
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    ref = make_agent(env_cls, seed=seed, affect="none")
    train = collect_transitions(env_name, 10, seed, ref)
    ref2 = make_agent(env_cls, seed=seed + 1, affect="none")
    heldout = collect_transitions(env_name, 10, seed + 100, ref2)[200:]
    rng = new_rng(seed + 5)
    shuffled = list(train)
    rng.shuffle(shuffled)
    a_ord = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, seed=seed + 2)
    for (o, a, o2, r) in train:
        a_ord.learn_transition(o, a, o2, r)
    a_shuf = ArchB(observation_space=space, n_actions=n_actions,
                   env_name=env_name, seed=seed + 2)
    for (o, a, o2, r) in shuffled:
        a_shuf.learn_transition(o, a, o2, r)
    e_ord = _mean([a_ord.eval_transition(o, a, o2)
                   for (o, a, o2, r) in heldout])
    e_shuf = _mean([a_shuf.eval_transition(o, a, o2)
                    for (o, a, o2, r) in heldout])
    kill = not (e_shuf > e_ord + 1e-9)
    return {"seed": seed, "mean_abs_e0_ordered_train": e_ord,
            "mean_abs_e0_shuffled_train": e_shuf, "delta": e_shuf - e_ord,
            "n_train": len(train), "n_heldout": len(heldout),
            "kill": bool(kill), "verdict_holds": not kill}


# ---------------------------------------------------------------- C1
def c1_one(seed):
    out = {}
    for env_name in ("delayed_reward", "grid_world"):
        agent = make_agent(ENVS_BY_NAME[env_name], seed=seed, affect="none")
        run_closed_loop(env_name, agent, 10, seed)
        errs = agent.agg["e0_abs"]
        n = len(errs)
        q = max(1, n // 5)
        early, late = _mean(errs[:q]), _mean(errs[-q:])
        out[env_name] = {"early": early, "late": late,
                         "decline": early - late, "n_ticks": n}
    holds = all(v["decline"] > 0 for v in out.values())
    return {"seed": seed, "envs": out, "holds": bool(holds),
            "verdict_holds": holds}


# ---------------------------------------------------------------- C2
def c2_one(seed):
    a_prec = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                        affect="none")
    eps_prec = run_closed_loop("changing_rule", a_prec, 10, seed)
    a_uni = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                       uniform_precision=True, affect="none")
    eps_uni = run_closed_loop("changing_rule", a_uni, 10, seed)
    m_prec = _mean([e["return"] for e in eps_prec])
    m_uni = _mean([e["return"] for e in eps_uni])
    kill_replicates = not (m_prec > m_uni + 1e-9)  # uniform beats precision
    return {"seed": seed, "mean_return_precision": m_prec,
            "mean_return_uniform": m_uni, "delta": m_prec - m_uni,
            "kill_replicates": bool(kill_replicates),
            "verdict_holds": bool(kill_replicates)}


# ---------------------------------------------------------------- C4
def c4_one(seed):
    out = {}
    for mode in ("active_inference", "random", "greedy"):
        agent = make_agent(ENVS_BY_NAME["pomaze"], seed=seed,
                           action_mode=mode, affect="none")
        run_closed_loop("pomaze", agent, 15, seed)
        errs = agent.agg["e0_abs"]
        n = len(errs)
        q = max(1, int(n * 0.15))
        out[mode] = {"IG": _mean(errs[:q]) - _mean(errs[-q:]), "n_ticks": n}
    for mode in out:
        agent = make_agent(ENVS_BY_NAME["pomaze"], seed=seed,
                           action_mode=mode, affect="none")
        eps = run_closed_loop("pomaze", agent, 15, seed)
        out[mode]["mean_return"] = _mean([e["return"] for e in eps])
    ig_ai = out["active_inference"]["IG"]
    kill_replicates = not (ig_ai > out["random"]["IG"] + 1e-9
                           and ig_ai > out["greedy"]["IG"] + 1e-9)
    return {"seed": seed, "modes": out,
            "kill_replicates": bool(kill_replicates),
            "verdict_holds": bool(kill_replicates)}


# ---------------------------------------------------------------- BAR
def bar_one(seed):
    logp = os.path.join(OUT_DIR, f"repro_EXP-AB-BAR.seed{seed}.predictions.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    intact = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                        log_path=logp, affect="none")
    eps_i = run_closed_loop("changing_rule", intact, 10, seed)
    if intact.plog:
        intact.plog.close()
    les = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed, lesion_l1=True,
                     affect="none")
    eps_l = run_closed_loop("changing_rule", les, 10, seed)

    def decline(errs):
        n = len(errs)
        q = max(1, n // 5)
        return _mean(errs[:q]) - _mean(errs[-q:])

    d_i, d_l = decline(intact.agg["e0_abs"]), decline(les.agg["e0_abs"])
    dr_i = decline(intact.agg["e0_reward"])
    dr_l = decline(les.agg["e0_reward"])
    r_i = _mean([e["return"] for e in eps_i])
    r_l = _mean([e["return"] for e in eps_l])
    fail = not (dr_i > 1e-9 and dr_l <= dr_i and r_i > 20.0)
    # original verdict: BAR NOT CLEARED (fail=True) -> reproduces if still fails
    return {"seed": seed, "intact_error_decline": d_i,
            "lesioned_error_decline": d_l,
            "intact_rewardchan_decline": dr_i,
            "lesioned_rewardchan_decline": dr_l,
            "intact_mean_return": r_i, "lesioned_mean_return": r_l,
            "chance_level": 20.0, "intact_beats_chance": r_i > 20.0,
            "cleared": not fail, "verdict_holds": bool(fail),
            "prediction_log": logp}


ONE_SEED_FNS = {"K1": k1_one, "K2": k2_one, "K3": k3_one, "K5": k5_one,
                "C1": c1_one, "C2": c2_one, "C4": c4_one, "BAR": bar_one}


def run_experiment(name):
    print(f"=== EXP-AB-{name} reproduction ===", flush=True)
    per_seed = []
    for seed in SEEDS[name]:
        try:
            r = ONE_SEED_FNS[name](seed)
            tag = ("HOLD" if r["verdict_holds"] else "FLIP")
            print(f"  seed={seed}: {tag} "
                  f"{json.dumps({k: v for k, v in r.items() if k not in ('episodes_learn', 'episodes_frozen', 'modes', 'envs', 'prediction_log')})[:220]}",
                  flush=True)
        except Exception as ex:
            r = {"seed": seed, "error": f"{type(ex).__name__}: {ex}",
                 "trace": traceback.format_exc(limit=5),
                 "verdict_holds": False}
            print(f"  seed={seed}: ERROR {r['error']}", flush=True)
        per_seed.append(r)
    holds = sum(1 for r in per_seed if r.get("verdict_holds"))
    repro = holds >= 4
    aggregate = {"seeds": SEEDS[name], "holds": holds, "n": len(per_seed),
                 "reproduces": repro}
    prereg = prereg_copy(name)
    prereg["conditions"]["reproduction_seeds"] = SEEDS[name]
    interp = (f"REPRODUCES — original verdict holds on {holds}/5 fresh seeds."
              if repro else
              f"FAILS TO REPRODUCE — original verdict holds on only "
              f"{holds}/5 fresh seeds. MAJOR FINDING.")
    lim = ("Multi-seed reproduction of the single-seed original. All RNG "
           "streams derive from the fresh primary seed; agent construction "
           "seeds = primary seed (originals used hardcoded agent seeds). "
           "Same environments, same code, same gates.")
    path = write_repro_receipt(f"EXP-AB-{name}", prereg, per_seed, aggregate,
                               interp, lim)
    print(f"EXP-AB-{name}: {holds}/5 -> "
          f"{'REPRODUCES' if repro else 'FAILS TO REPRODUCE'}\n  {path}",
          flush=True)
    return repro


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL",
                    choices=sorted(ONE_SEED_FNS) + ["ALL"])
    args = ap.parse_args(argv)
    names = sorted(ONE_SEED_FNS) if args.exp == "ALL" else [args.exp]
    summary = {}
    for name in names:
        summary[name] = run_experiment(name)
    print("\n=== REPRODUCTION SUMMARY ===")
    for name, v in summary.items():
        print(f"B-{name}: {'REPRODUCES' if v else 'FAILS TO REPRODUCE'}")


if __name__ == "__main__":
    main()
