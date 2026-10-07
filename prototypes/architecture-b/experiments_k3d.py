"""EXP-AB-K3D — K3B |e0|-channel characterization (mechanism: memorization vs dynamics).

Background: EXP-AB-K3B's reward-channel headline (+0.0602) reversed on fresh
seeds (repro5: -0.0319); the surviving hierarchy signal rides |e0|
(+0.0079 seed-mean, 4/5 positive, ~2.5x K3 scale). Recorded as NR-B-009.

This experiment characterizes the surviving |e0| channel on four envs
(changing_rule, delayed_reward, delayed_multistep, compositional_rule) with
confidence intervals, and asks the mechanism question on the two flip envs:
does the |e0| reduction come from the L1 D[ctx] tables memorizing the old
phase's per-context contingency means (advantage vanishes/reverses
post-flip — the K3C signature), or from genuine context-dependent dynamics
(advantage survives the flip, in rule-invariant channels)?

PREREGISTERED (2026-10-07, before run; full JSON:
flesh-pits/experiments/preregistration_K3B_E0.json)
- characterization: descriptive success criteria S1/S2/S3, no kill gate.
- metric: per seed/segment D_e0 = mean|e0|_lesioned - mean|e0|_intact;
  per-channel D_c; 95% t-CI (df=3).
- instruments verbatim where possible: reference collection via
  collect_transitions / make_agent (experiments.py); scripted multistep via
  experiments_k3c.collect_scripted_multistep; training via the
  experiments_phase4._train_eval_ab path (ArchB + learn_transition open-loop,
  no retrieval correction/affect); eval via the model's predict_next path
  (exactly the computation eval_transition performs) with per-trial
  per-channel errors recorded additionally.
- seeds: [76101, 76102, 76103, 76104] — fresh, zero overlap with any prior
  lab seed (checked before sealing).
- interpretation signatures: MEMORIZATION / DYNAMICS / MIXED (preregistered).

Usage:
    python3 experiments_k3d.py
"""

import copy
import datetime
import hashlib
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP_EXP = os.path.join(HERE, "..", "..", "..", "flesh-pits", "experiments")
sys.path.insert(0, FP_EXP)

from env_interface import (  # noqa: E402
    config_hash, derive_seed, obs_to_vector,
)
from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, collect_transitions, write_receipt, _mean,
)
from agent import ArchB  # noqa: E402
from experiments_k3c import collect_scripted_multistep  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "receipts")
LANE_RECEIPTS_DIR = os.path.abspath(os.path.join(FP_EXP, "..", "receipts"))

SEEDS = [76101, 76102, 76103, 76104]
AGENT_SEED_BASE = 76120   # train/eval agent init seed base (paired arms)
REF_SEED_BASE = 76210     # reference agent seed bases (per env below)
T_CRIT_95_DF3 = 3.1824    # Student's t, two-sided 95%, df=3

# rule-contingent vs rule-invariant obs channels (preregistered)
CONTINGENT = {
    "changing_rule": {"last_reward"},
    "compositional_rule": {"last_reward"},
    "delayed_reward": set(),
    "delayed_multistep": set(),
}


def channel_map(space):
    """Field -> list of vector indices (declared field order, one-hot)."""
    cmap, idx = {}, 0
    for field, spec in space.items():
        t = spec["type"]
        if t == "scalar":
            cmap[field] = [idx]
            idx += 1
        elif t == "vector":
            n = len(spec.get("values", []))
            cmap[field] = list(range(idx, idx + n))
            idx += n
        elif t == "categorical":
            n = len(spec["values"])
            cmap[field] = list(range(idx, idx + n))
            idx += n
    return cmap


def _train_agent(transitions, lesion_l1, seed, env_name):
    """Verbatim training path of experiments_phase4._train_eval_ab."""
    cls = ENVS_BY_NAME[env_name]
    space = cls().observation_space()
    n_actions = cls().action_space()["n"]
    agent = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, lesion_l1=lesion_l1, seed=seed,
                  affect="none")
    for (o, a, o2, r) in transitions:
        agent.learn_transition(o, a, o2, r)
    return agent


def _eval_channels(agent, transitions):
    """Per-trial per-channel |err| via the model's predict_next path —
    exactly the computation eval_transition performs (|e0| = L2 norm)."""
    per_trial_l2 = []
    per_trial_chan = []
    for (o, a, o2, r) in transitions:
        ctx = agent.context_key(o)
        xhat = agent.model.predict_next(o, a, ctx)
        errs = [abs(x - y) for x, y in zip(o2, xhat)]
        per_trial_l2.append(math.sqrt(sum(e * e for e in errs)))
        per_trial_chan.append(errs)
    return per_trial_l2, per_trial_chan


def _summarize_segment(intact_trials, les_trials, cmap):
    """intact_trials / les_trials: (l2_list, chan_list). Returns dict with
    per-trial L2 means and per-field channel means."""
    i_l2, i_ch = intact_trials
    l_l2, l_ch = les_trials
    d_l2 = [l - i for l, i in zip(l_l2, i_l2)]
    fields = {}
    n = len(i_l2)
    for field, idxs in cmap.items():
        i_m = _mean([sum(t[j] for j in idxs) / len(idxs) for t in i_ch])
        l_m = _mean([sum(t[j] for j in idxs) / len(idxs) for t in l_ch])
        fields[field] = {"intact": i_m, "lesioned": l_m, "D": l_m - i_m}
    return {
        "n_trials": n,
        "intact_mean_e0": _mean(i_l2),
        "lesioned_mean_e0": _mean(l_l2),
        "D_e0": _mean(d_l2),
        "per_trial_D_e0": d_l2,
        "fields": fields,
    }


def ci95(xs):
    """95% t-CI (df=n-1). Returns (mean, lo, hi)."""
    n = len(xs)
    if n < 2:
        m = _mean(xs)
        return m, m, m
    m = statistics.fmean(xs)
    s = statistics.stdev(xs)
    half = T_CRIT_95_DF3 * s / math.sqrt(n) if n == 4 else (
        12.7062 * s / math.sqrt(n) if n == 2 else 4.3027 * s / math.sqrt(n))
    return m, m - half, m + half


# ---------------------------------------------------------------------------
# per-env runners (reference collection faithful to K3B/K3C)
# ---------------------------------------------------------------------------

def run_changing_rule(seed, si):
    """45 eps; train 0-1499 (phases 0-7); stable = eps 38-39 (phase 7);
    post-flip = eps 40-44 (phase 8)."""
    ref = make_agent(ENVS_BY_NAME["changing_rule"], seed=REF_SEED_BASE + si,
                     affect="none")
    all_t = collect_transitions("changing_rule", 45, seed, ref)
    train, stable, postflip = all_t[:1500], all_t[1520:1600], all_t[1600:1800]
    if not stable or not postflip:
        return {"void": True, "reason": "G0a: empty held-out segment"}
    ag = AGENT_SEED_BASE + si
    intact = _train_agent(train, False, ag, "changing_rule")
    les = _train_agent(train, True, ag, "changing_rule")
    space = ENVS_BY_NAME["changing_rule"]().observation_space()
    cmap = channel_map(space)
    return {
        "seed": seed, "n_train": len(train),
        "n_contexts_intact": intact.model.n_contexts,
        "stable": _summarize_segment(_eval_channels(intact, stable),
                                     _eval_channels(les, stable), cmap),
        "postflip": _summarize_segment(_eval_channels(intact, postflip),
                                       _eval_channels(les, postflip), cmap),
    }


def run_compositional_rule(seed, si):
    """45 eps; train 0-1519 (phases 0-7); stable = eps 38-39 (phase 7);
    post-flip = eps 40-44 (phase 8). 80 transitions shorter than K3C."""
    ref = make_agent(ENVS_BY_NAME["compositional_rule"],
                     seed=REF_SEED_BASE + 10 + si, affect="none")
    all_t = collect_transitions("compositional_rule", 45, seed, ref)
    train, stable, postflip = all_t[:1520], all_t[1520:1600], all_t[1600:1800]
    if not stable or not postflip:
        return {"void": True, "reason": "G0a: empty held-out segment"}
    ag = AGENT_SEED_BASE + si
    intact = _train_agent(train, False, ag, "compositional_rule")
    les = _train_agent(train, True, ag, "compositional_rule")
    space = ENVS_BY_NAME["compositional_rule"]().observation_space()
    cmap = channel_map(space)
    return {
        "seed": seed, "n_train": len(train),
        "n_contexts_intact": intact.model.n_contexts,
        "stable": _summarize_segment(_eval_channels(intact, stable),
                                     _eval_channels(les, stable), cmap),
        "postflip": _summarize_segment(_eval_channels(intact, postflip),
                                       _eval_channels(les, postflip), cmap),
    }


def run_delayed_reward(seed, si):
    """60 eps; train first 600; held-out last 120 (K3B-arm-B-faithful)."""
    ref = make_agent(ENVS_BY_NAME["delayed_reward"],
                     seed=REF_SEED_BASE + 20 + si, affect="none")
    all_t = collect_transitions("delayed_reward", 60, seed, ref)
    train, heldout = all_t[:600], all_t[-120:]
    if not heldout:
        return {"void": True, "reason": "G0a: empty held-out"}
    ag = AGENT_SEED_BASE + si
    intact = _train_agent(train, False, ag, "delayed_reward")
    les = _train_agent(train, True, ag, "delayed_reward")
    space = ENVS_BY_NAME["delayed_reward"]().observation_space()
    cmap = channel_map(space)
    return {
        "seed": seed, "n_train": len(train),
        "n_contexts_intact": intact.model.n_contexts,
        "heldout": _summarize_segment(_eval_channels(intact, heldout),
                                      _eval_channels(les, heldout), cmap),
    }


def run_delayed_multistep(seed, si):
    """30 eps, scripted reference; train eps 0-19; held-out eps 20-29."""
    eps = collect_scripted_multistep(30, seed)
    train = [t for ep in eps[:20] for t in ep]
    heldout = [t for ep in eps[20:] for t in ep]
    if not heldout:
        return {"void": True, "reason": "G0a: empty held-out"}
    ag = AGENT_SEED_BASE + si
    intact = _train_agent(train, False, ag, "delayed_multistep")
    les = _train_agent(train, True, ag, "delayed_multistep")
    space = ENVS_BY_NAME["delayed_multistep"]().observation_space()
    cmap = channel_map(space)
    return {
        "seed": seed, "n_train": len(train),
        "n_contexts_intact": intact.model.n_contexts,
        "heldout": _summarize_segment(_eval_channels(intact, heldout),
                                      _eval_channels(les, heldout), cmap),
    }


K3D_PREREG = {
    "experiment_id": "EXP-AB-K3D",
    "task": "K3B |e0|-channel characterization (mechanism: memorization "
            "vs dynamics)",
    "hypothesis": "DESCRIPTIVE: characterize D_e0 per env with 95% CIs; "
                  "decompose stable vs post-flip on the two flip envs.",
    "null": "D_e0 <= 0 seed-mean on all 4 envs — the surviving channel "
            "is dead too.",
    "metric": "D_e0 = lesioned - intact mean|e0| per seed/segment; per-"
              "channel D_c; 95% t-CI (df=3).",
    "baseline": "arch_b lesion_l1=True (L0-only), paired streams/seeds.",
    "ablation": "L1 top-down path (lesion_l1).",
    "procedure": "Open-loop learn_transition training (K3B/K3C-faithful); "
                 "per-trial per-channel eval via the predict_next path. "
                 "Full: flesh-pits/experiments/preregistration_K3B_E0.json.",
    "seeds": list(SEEDS),
    "conditions": {
        "agent": "arch_b v1", "affect": "none", "contract": "1.0.0",
        "changing_rule": "45 eps, train 0-1499, stable 1520-1599 (ph7), "
                         "postflip 1600-1799 (ph8); ref base 76210",
        "compositional_rule": "45 eps, train 0-1519, stable 1520-1599 "
                              "(ph7), postflip 1600-1799 (ph8); ref base "
                              "76220",
        "delayed_reward": "60 eps, train first 600, held-out last 120; "
                          "ref base 76230",
        "delayed_multistep": "30 eps scripted, train eps 0-19, held-out "
                             "eps 20-29",
        "agent_init_base": "76120+si (paired)",
    },
    "decision_rule": "CHARACTERIZATION (S1/S2/S3 descriptive; signatures "
                     "MEMORIZATION/DYNAMICS/MIXED). No kill gate.",
}


def _env_summary(per_seed, seg_key):
    ds = [p[seg_key]["D_e0"] for p in per_seed]
    m, lo, hi = ci95(ds)
    return {
        "seed_mean_D_e0": m, "ci95_lo": lo, "ci95_hi": hi,
        "n_positive": sum(1 for d in ds if d > 0),
        "per_seed_D_e0": ds,
        "per_seed": per_seed,
    }


def _mechanism(env_name, per_seed):
    """Preregistered signature classification (flip envs only)."""
    if env_name not in ("changing_rule", "compositional_rule"):
        return {"signature": "N/A (no flip structure)"}
    ds = [p["stable"]["D_e0"] for p in per_seed]
    dp = [p["postflip"]["D_e0"] for p in per_seed]
    ms, _, _ = ci95(ds)
    mp, _, _ = ci95(dp)
    cont = CONTINGENT[env_name]
    inv = [f for f in per_seed[0]["stable"]["fields"] if f not in cont]
    def chan_means(seg, fields):
        return {f: _mean([p[seg]["fields"][f]["D"] for p in per_seed])
                for f in fields}
    out = {
        "stable_seed_mean_D_e0": ms, "postflip_seed_mean_D_e0": mp,
        "per_seed_stable": ds, "per_seed_postflip": dp,
        "contingent_channels": sorted(cont),
        "stable_contingent_D": chan_means("stable", sorted(cont)),
        "postflip_contingent_D": chan_means("postflip", sorted(cont)),
        "stable_invariant_D": chan_means("stable", inv),
        "postflip_invariant_D": chan_means("postflip", inv),
    }
    if mp <= 0 and ms > 0:
        out["signature"] = "MEMORIZATION"
    elif mp > 0:
        out["signature"] = "DYNAMICS"
    else:
        out["signature"] = "MIXED"
    return out


def g0b_determinism_check():
    """G0b: recompute first-seed changing_rule intact arm through the
    identical helper; per-trial L2 must match to 1e-12."""
    s, si = SEEDS[0], 0
    r1 = run_changing_rule(s, si)
    r2 = run_changing_rule(s, si)
    a = r1["stable"]["per_trial_D_e0"]
    b = r2["stable"]["per_trial_D_e0"]
    ok = (len(a) == len(b)
          and all(abs(x - y) <= 1e-12 for x, y in zip(a, b)))
    return ok


def main():
    prereg = copy.deepcopy(K3D_PREREG)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    runners = {
        "changing_rule": run_changing_rule,
        "delayed_reward": run_delayed_reward,
        "delayed_multistep": run_delayed_multistep,
        "compositional_rule": run_compositional_rule,
    }
    out = {}
    for env_name, runner in runners.items():
        per_seed = []
        for si, s in enumerate(SEEDS):
            print(f"--- {env_name} seed {s} ({si + 1}/{len(SEEDS)}) ---",
                  flush=True)
            r = runner(s, si)
            per_seed.append(r)
            if r.get("void"):
                print(f"  VOID: {r['reason']}", flush=True)
            else:
                segs = [k for k in r if k in ("stable", "postflip",
                                              "heldout")]
                print("  " + " ".join(
                    f"{k}:D_e0={r[k]['D_e0']:+.4f}"
                    for k in segs), flush=True)
        seg_keys = [k for k in ("stable", "postflip", "heldout")
                    if k in per_seed[0] and not per_seed[0].get("void")]
        out[env_name] = {
            seg: _env_summary(per_seed, seg) for seg in seg_keys
        }
        out[env_name]["mechanism"] = _mechanism(env_name, per_seed)
        out[env_name]["void"] = any(p.get("void") for p in per_seed)

    g0b = g0b_determinism_check()
    print(f"G0b determinism check: {'PASS' if g0b else 'FAIL'}", flush=True)
    out["G0b_determinism"] = g0b

    # -- interpretation from the preregistered signatures --
    lines = []
    for env_name in runners:
        e = out[env_name]
        if e.get("void"):
            lines.append(f"{env_name}: VOID.")
            continue
        for seg in ("stable", "postflip", "heldout"):
            if seg in e:
                s = e[seg]
                lines.append(
                    f"{env_name}/{seg}: D_e0 seed-mean "
                    f"{s['seed_mean_D_e0']:+.4f} "
                    f"95%CI [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}], "
                    f"positive {s['n_positive']}/4 seeds.")
        mech = e["mechanism"]
        if mech["signature"] not in ("N/A (no flip structure)",):
            lines.append(
                f"{env_name} mechanism: {mech['signature']} "
                f"(stable {mech['stable_seed_mean_D_e0']:+.4f}, "
                f"postflip {mech['postflip_seed_mean_D_e0']:+.4f}).")
    sigs = {env: out[env]["mechanism"]["signature"]
            for env in ("changing_rule", "compositional_rule")}
    interp = ("EXP-AB-K3D |e0|-channel characterization (4 fresh seeds, "
              "preregistered). " + " ".join(lines)
              + f" Flip-env signatures: {sigs}.")
    lim = ("Open-loop training isolates the weight/precision machinery "
           "(K3/K3B/K3C-faithful); closed-loop control effects factored out "
           "by design. compositional_rule train is 80 transitions shorter "
           "than K3C's split (preregistered). Per-channel |err| uses the "
           "exact predict_next computation eval_transition performs. CIs "
           "(95 pct) are t-based, df=3 (wide by construction at n=4). "
           "G0b determinism: %s." % ("PASS" if g0b else "FAIL"))

    # -- arch-b detail receipt (K3-lineage chain link to EXP-AB-K3C) --
    arch_path = write_receipt("EXP-AB-K3D", prereg, out, interp, lim)
    with open(arch_path) as f:
        body = json.load(f)
    k3c_path = os.path.join(RECEIPTS_DIR, "EXP-AB-K3C.json")
    with open(k3c_path, "rb") as f:
        prev_hash = hashlib.sha256(f.read()).hexdigest()
    body["prev_receipt_hash"] = prev_hash
    canonical = json.dumps(body, indent=2, sort_keys=True)
    body["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    with open(arch_path, "w") as f:
        json.dump(body, f, indent=2, sort_keys=True)

    # -- lane summary receipt (joins the harness hash chain) --
    sys.path.insert(0, FP_EXP)
    from harness import write_receipt as lane_write_receipt  # noqa: E402
    lane_result = {
        "experiment_id": "EXP-AB-K3D",
        "config": {"envs": ["changing_rule v1.0.0",
                            "delayed_reward v1.0.0",
                            "delayed_multistep v1.0.0",
                            "compositional_rule v1.0.0"],
                   "seeds": SEEDS, "lesion": "lesion_l1",
                   "instrument": "K3B open-loop + per-channel predict_next "
                                 "eval"},
        "config_hash": config_hash(prereg["conditions"]),
        "primary_seed": "76101-76104",
        "started_utc": started,
        "episodes": {env: {"segments": list(out[env].keys())}
                     for env in runners},
        "summary": {
            "verdict": "CHARACTERIZATION COMPLETE",
            "per_env": {
                env: {
                    seg: {
                        "seed_mean_D_e0": out[env][seg]["seed_mean_D_e0"],
                        "ci95": [out[env][seg]["ci95_lo"],
                                 out[env][seg]["ci95_hi"]],
                        "n_positive": out[env][seg]["n_positive"],
                    }
                    for seg in out[env] if seg in ("stable", "postflip",
                                                   "heldout")
                } for env in runners
            },
            "mechanism_signatures": sigs,
            "G0b_determinism": g0b,
            "detail_receipt": ("prototypes/architecture-b/receipts/"
                               "EXP-AB-K3D.json"),
        },
    }
    lane_path = lane_write_receipt(
        lane_result, LANE_RECEIPTS_DIR,
        hypothesis=prereg["hypothesis"], null=prereg["null"],
        preregistered_metric=prereg["metric"], baseline=prereg["baseline"],
        conditions=json.dumps(prereg["conditions"], sort_keys=True),
        interpretation=interp, limitations=lim)
    print(f"\nDONE\n  arch receipt: {arch_path}\n  lane receipt: {lane_path}")


if __name__ == "__main__":
    main()
