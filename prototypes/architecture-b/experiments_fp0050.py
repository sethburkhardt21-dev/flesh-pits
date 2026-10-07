"""EXP-FP-0050 — hierarchy beyond the two envs (FINAL_HANDOFF §9 item 7).

FINAL_HANDOFF §9 item 7 / gap #3: "Hierarchy beyond the two envs (B gap #3)
— harder tasks (delayed multi-step, compositional rules) with reward-channel
instruments; preregister the lesion gap."

K3B's hierarchy signal (L1 top-down lesion; arm-A |rerr| gap +0.0602, 3/3)
stands only on changing_rule + delayed_reward. K3C tested the two harder
tasks with prediction-error instruments (|e0|/|rerr|, open-loop) -> NO
EFFECT (task1 D_e0 -0.0470, task2 D_rerr -0.2684 seed-mean). K3E tested
whether the surviving prediction signal is load-bearing for CONTROL on
delayed_reward -> prediction-only decoration.

This experiment is the remaining cell: CLOSED-LOOP CONTROL lesion gap
(performance = mean episode return, intact vs lesion_l1, learning on) on
the two harder tasks — delayed_multistep and compositional_rule —
preregistered with a solvability sanity gate and a consistent-sign
decision rule (per the mandate: gap consistent in sign across 4 fresh
seeds; unlesioned agent must solve the task).

Usage:
    python3 experiments_fp0050.py --pilot     # solvability pilot only
    python3 experiments_fp0050.py             # full preregistered run
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP_EXP = os.path.abspath(os.path.join(HERE, "..", "..", "experiments"))
sys.path.insert(0, FP_EXP)

from env_interface import config_hash, derive_seed  # noqa: E402
from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, run_closed_loop, write_receipt, _mean,
)
from agent import ArchB  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "receipts")
LANE_RECEIPTS_DIR = os.path.abspath(os.path.join(FP_EXP, "..", "receipts"))

SEEDS = [78101, 78102, 78103, 78104]
AGENT_SEED_BASE = 78120   # paired intact/lesioned agent init seeds
N_EPS = 30                # closed-loop episodes per arm per seed
TASKS = ["delayed_multistep", "compositional_rule"]

# Preregistered sanity-gate thresholds (set from the pilot, sealed before
# the main run): seed-mean unlesioned return must clear these.
#  delayed_multistep: pilot intact 0.148 (12ep) / 0.045 (30ep, 0/30 +1.0
#    hits), random 0.217 -> bar 0.50 requires genuine +1.0 hits.
#  compositional_rule: pilot intact 28.75, random 23.0 (analytic 20.0) ->
#    bar 24.0 (60% correct).
SOLVE_BAR = {
    "delayed_multistep": 0.50,
    "compositional_rule": 24.0,
}


def _space(env_name):
    cls = ENVS_BY_NAME[env_name]
    return cls().observation_space(), cls().action_space()["n"]


def run_task(env_name, seed, si, n_episodes, random_baseline=False):
    """Closed-loop lesion arm pair. Paired streams + paired agent inits.

    Returns dict with per-episode returns for intact / lesioned arms and
    the lesion gap G = mean(intact) - mean(lesioned) (performance units;
    positive = hierarchy helps control).
    """
    space, n_actions = _space(env_name)
    kw = dict(observation_space=space, n_actions=n_actions,
              env_name=env_name, action_mode="active_inference",
              affect="none")
    intact = ArchB(lesion_l1=False, seed=AGENT_SEED_BASE + si, **kw)
    les = ArchB(lesion_l1=True, seed=AGENT_SEED_BASE + si, **kw)
    ep_i = run_closed_loop(env_name, intact, n_episodes, seed)
    ep_l = run_closed_loop(env_name, les, n_episodes, seed)
    ri = [e["return"] for e in ep_i]
    rl = [e["return"] for e in ep_l]
    out = {
        "env": env_name,
        "seed": seed,
        "n_episodes": n_episodes,
        "returns_intact": ri,
        "returns_lesioned": rl,
        "mean_intact": _mean(ri),
        "mean_lesioned": _mean(rl),
        "stdev_intact": statistics.pstdev(ri) if len(ri) > 1 else 0.0,
        "stdev_lesioned": statistics.pstdev(rl) if len(rl) > 1 else 0.0,
        "G": _mean(ri) - _mean(rl),
        "n_contexts_intact": intact.model.n_contexts,
        "n_contexts_lesioned": les.model.n_contexts,
    }
    if random_baseline:
        rnd = ArchB(action_mode="random", affect="none",
                    observation_space=space, n_actions=n_actions,
                    env_name=env_name, seed=AGENT_SEED_BASE + si)
        ep_r = run_closed_loop(env_name, rnd, n_episodes, seed)
        rr = [e["return"] for e in ep_r]
        out["returns_random"] = rr
        out["mean_random"] = _mean(rr)
    return out


def g0b_determinism_check():
    """G0b: recompute (first task, first seed) intact arm through the
    identical helper; per-episode returns must match to 1e-12."""
    r1 = run_task(TASKS[0], SEEDS[0], 0, N_EPS)
    r2 = run_task(TASKS[0], SEEDS[0], 0, N_EPS)
    a, b = r1["returns_intact"], r2["returns_intact"]
    return (len(a) == len(b)
            and all(abs(x - y) <= 1e-12 for x, y in zip(a, b)))


FP0050_PREREG = {
    "experiment_id": "EXP-FP-0050",
    "task": ("Hierarchy beyond the two envs: closed-loop control lesion "
             "gap on delayed_multistep + compositional_rule."),
    "hypothesis": ("B's hierarchy (L1 top-down path / reward-channel "
                   "contingency) earns its keep for CONTROL on harder "
                   "tasks: intact beats L1-lesioned on closed-loop mean "
                   "return, consistently across seeds, on both "
                   "delayed_multistep and compositional_rule."),
    "null": ("No consistent control gap: the lesion gap is ~0 or "
             "inconsistent in sign — the hierarchy does not earn its keep "
             "for control on harder tasks (K3C's prediction-error null "
             "extends to control)."),
    "metric": ("Per task, per seed: G = mean_return(intact) - "
               "mean_return(lesioned), closed-loop, learning on, paired "
               "streams + paired agent inits. 4 fresh seeds {78101..78104}. "
               "EARNS_KEEP(task) iff G_s > 0 on all 4 seeds AND the "
               "solvability sanity gate passes."),
    "baseline": "ArchB lesion_l1=True (L0-only), closed-loop, learning on.",
    "ablation": "L1 top-down path (lesion_l1) — the K3B/K3C/K3D/K3E lesion.",
    "sanity_gate": ("Per task: seed-mean mean_return(intact) >= SOLVE_BAR, "
                    "sealed pre-run from the pilot: delayed_multistep=0.50 "
                    "(pilot intact 0.148/0.045, random 0.217 — bar requires "
                    "genuine +1.0 hits), compositional_rule=24.0 (pilot "
                    "intact 28.75, random 23.0 / analytic 20.0 — bar is 60% "
                    "correct). Sanity FAIL -> task VOID for the lesion-gap "
                    "question (cannot measure a control gap on an unsolved "
                    "task); recorded as a bound, not a null."),
    "pilot_record": ("Throwaway seeds 78111 (delayed_multistep) / 78112 "
                     "(compositional_rule); agent inits 78210/78211; intact "
                     "+ lesioned + random arms, 12 episodes; extended "
                     "30-episode intact/lesioned check on delayed_multistep "
                     "seed 78111 (agent init 78211). Zero overlap with main "
                     "seeds {78101..78104} / inits {78120..78123}."),
    "procedure": ("Full: flesh-pits/experiments/preregistration_EXP-FP-0050.json. "
                  "Pilot (above) established solvability per task and set "
                  "SOLVE_BAR; main run uses 4 fresh seeds with zero overlap."),
    "seeds": list(SEEDS),
    "conditions": {
        "agent": "arch_b v1", "affect": "none",
        "action_mode": "active_inference", "learning": "on (both arms)",
        "envs": {t: ENVS_BY_NAME[t].VERSION for t in TASKS},
        "n_episodes": N_EPS,
        "agent_init_base": "78120+si (paired intact/lesioned)",
        "instrument": "run_closed_loop verbatim (prototypes/architecture-b/experiments.py)",
    },
    "decision_rule": ("Per task: EARNS_KEEP iff G>0 on all 4 seeds AND "
                      "sanity gate passes. Else NULL (no consistent gap) "
                      "or VOID (sanity fail)."),
}


def pilot():
    print("=== EXP-FP-0050 PILOT (throwaway seeds; informs preregistration) ===",
          flush=True)
    for ti, task in enumerate(TASKS):
        pseed = 78111 + ti
        r = run_task(task, pseed, 90 + ti, 12, random_baseline=True)
        print(f"{task}: seed={pseed}", flush=True)
        print(f"  intact   mean={r['mean_intact']:+.4f} "
              f"(sd {r['stdev_intact']:.4f}) n_ctx={r['n_contexts_intact']}",
              flush=True)
        print(f"  lesioned mean={r['mean_lesioned']:+.4f} "
              f"(sd {r['stdev_lesioned']:.4f}) n_ctx={r['n_contexts_lesioned']}",
              flush=True)
        print(f"  random   mean={r['mean_random']:+.4f}", flush=True)
        print(f"  pilot G={r['G']:+.4f}", flush=True)
        print(f"  intact returns: {[round(x,3) for x in r['returns_intact']]}",
              flush=True)


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    per_task = {}
    for ti, task in enumerate(TASKS):
        per_seed = []
        for si, s in enumerate(SEEDS):
            print(f"--- {task} seed {s} ({si + 1}/{len(SEEDS)}) ---",
                  flush=True)
            try:
                # agent init offset per task keeps cross-task inits distinct
                r = run_task(task, s, si + 10 * ti, N_EPS)
                print(f"  intact={r['mean_intact']:+.4f} "
                      f"lesioned={r['mean_lesioned']:+.4f} "
                      f"G={r['G']:+.4f} ctx={r['n_contexts_intact']}",
                      flush=True)
                per_seed.append({"seed": s, "void": False, **r})
            except Exception as e:  # noqa: BLE001 — G0d: VOID the seed
                print(f"  VOID: {e!r}\n{traceback.format_exc()}", flush=True)
                per_seed.append({"seed": s, "void": True,
                                 "reason": f"crash: {e!r}"})
        per_task[task] = per_seed

    g0b = g0b_determinism_check()
    print(f"G0b determinism check: {'PASS' if g0b else 'FAIL'}", flush=True)

    verdicts = {}
    summary = {}
    for task in TASKS:
        valid = [p for p in per_task[task] if not p.get("void")]
        gs = [p["G"] for p in valid]
        mi = [p["mean_intact"] for p in valid]
        n = len(valid)
        bar = SOLVE_BAR[task]
        sanity = (bar is not None and n == 4
                  and _mean(mi) >= bar)
        earns = sanity and n == 4 and all(g > 0 for g in gs)
        summary[task] = {
            "n_valid": n,
            "per_seed_G": gs,
            "seed_mean_G": _mean(gs) if gs else 0.0,
            "per_seed_mean_intact": mi,
            "seed_mean_intact": _mean(mi) if mi else 0.0,
            "solve_bar": bar,
            "sanity_pass": bool(sanity),
            "n_positive": sum(1 for g in gs if g > 0),
            "earns_keep": bool(earns),
        }
        if not sanity:
            verdicts[task] = ("VOID: solvability sanity gate failed — the "
                              "unlesioned agent does not solve this task, "
                              "so no control lesion gap can be measured.")
        elif earns:
            verdicts[task] = ("EARNS_KEEP: lesion gap positive on all 4 "
                              "seeds with the sanity gate passed — the "
                              "hierarchy earns its keep for control on "
                              "this harder task.")
        else:
            verdicts[task] = ("NULL: no consistent control gap — the "
                              "hierarchy does not earn its keep for "
                              "control on this harder task.")
        print(f"{task}: seed-mean G={summary[task]['seed_mean_G']:+.4f}, "
              f"positive {summary[task]['n_positive']}/{n}, "
              f"sanity={'PASS' if sanity else 'FAIL'} -> "
              f"{'EARNS_KEEP' if earns else ('VOID' if not sanity else 'NULL')}",
              flush=True)

    out = {
        "per_task": per_task,
        "summary": summary,
        "G0b_determinism": g0b,
        "verdicts": verdicts,
        "solve_bar": dict(SOLVE_BAR),
    }
    interp = ("EXP-FP-0050 closed-loop control lesion gaps (4 fresh seeds, "
              "preregistered). " + " ".join(
                  f"{t}: per-seed G={summary[t]['per_seed_G']}; "
                  f"seed-mean {summary[t]['seed_mean_G']:+.4f}; "
                  f"sanity {'PASS' if summary[t]['sanity_pass'] else 'FAIL'}; "
                  f"{verdicts[t]}" for t in TASKS))
    lim = ("Closed-loop ArchB (active_inference, affect='none'), learning "
           "on, 30 episodes/arm/seed. Paired env streams + paired agent "
           "inits within each seed. G0b determinism: %s. "
           "Sanity bars sealed pre-run from the pilot. "
           "CONSCIOUSNESS: UNRESOLVED — return gaps are mechanisms, not "
           "subjects." % ("PASS" if g0b else "FAIL"))

    # -- arch-b detail receipt (K-lineage chain link to EXP-AB-K3E) --
    prereg = copy.deepcopy(FP0050_PREREG)
    arch_path = write_receipt("EXP-FP-0050", prereg, out, interp, lim)
    with open(arch_path) as f:
        body = json.load(f)
    k3e_path = os.path.join(RECEIPTS_DIR, "EXP-AB-K3E.json")
    with open(k3e_path, "rb") as f:
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
        "experiment_id": "EXP-FP-0050",
        "config": {"envs": {t: ENVS_BY_NAME[t].VERSION for t in TASKS},
                   "seeds": SEEDS,
                   "arms": ["intact vs lesion_l1 closed-loop, learning on"],
                   "n_episodes": N_EPS,
                   "instrument": "run_closed_loop verbatim",
                   "solve_bar": dict(SOLVE_BAR)},
        "config_hash": config_hash(prereg["conditions"]),
        "primary_seed": "78101-78104",
        "started_utc": started,
        "episodes": {"n_seeds": 4, "n_tasks": len(TASKS),
                     "voided": [p["seed"] for t in TASKS
                                for p in per_task[t] if p.get("void")]},
        "summary": {"verdicts": verdicts, "per_task": summary,
                    "G0b_determinism": g0b,
                    "detail_receipt": ("prototypes/architecture-b/receipts/"
                                       "EXP-FP-0050.json")},
    }
    lane_path = lane_write_receipt(
        lane_result, LANE_RECEIPTS_DIR,
        hypothesis=prereg["hypothesis"], null=prereg["null"],
        preregistered_metric=prereg["metric"], baseline=prereg["baseline"],
        conditions=json.dumps(prereg["conditions"], sort_keys=True),
        interpretation=interp, limitations=lim)
    print(f"\nDONE\n  arch receipt: {arch_path}\n  lane receipt: {lane_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        pilot()
    else:
        main()
