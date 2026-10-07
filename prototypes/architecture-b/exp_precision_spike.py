"""EXP-FP-0008 — precision-explosion characterization: estimated vs uniform
reward head on rare terminal spikes (C2 pathology family follow-up).

Preregistered spec sealed in experiments/preregistration_EXP-FP-0008.json
AND experiments/EXPERIMENT_REGISTRY.md before any code was written; this
file is the executor, not the spec.

Background: EXP-AB-K3C task-1 pilot (throwaway seed 99901,
delayed_multistep): the lesioned model's estimated-precision reward head
exploded on rare terminal +1.0 spikes (5/600 train transitions) —
b_r=0.31, max|w_r|=0.73, rhat=1.27 on a 0.02 tick; held-out
mean|rerr|=0.8256 vs intact 0.2809; lesioned+uniform control stayed sane
(b_r=0.018, mean|rerr|=0.10). Mechanism hypothesis: rare spikes in a ~0
reward sea lower rerr variance -> estimated piR stays high -> massive
per-spike w_r updates (d = eta_r*piR*rerr*f) with no absorber. Second
independent sighting of the C2 precision pathology family (C2 kill was
FRAGILE 3/5).

H: the estimated-precision reward head systematically misbehaves on rare
   high-reward events — across tasks with rare terminal spikes, intact
   ArchB with estimated precision shows the explosion signature on
   terminal-spike trials while the uniform-precision control stays sane;
   on a dense-reward task the signature is absent.
H0: no systematic estimated-vs-uniform difference on terminal-spike
   trials across tasks.

Metric (EXACT): per task per seed, on held-out terminal-spike trials
(reward >= 0.5): explosion signature iff
mean|rerr|_estimated_terminal >= 3 * mean|rerr|_uniform_terminal AND
mean|rerr|_estimated_terminal > 0.3. Task signature iff >= 3/4 seeds.

Verdict mapping (CHARACTERIZATION, not a kill): SYSTEMATIC iff >= 2 of 3
tasks show the task signature; TASK-LOCAL iff exactly 1; ABSENT iff 0.

Procedure (K3B-instrument-faithful): per task per seed, the symbolic
reference baseline collects a transition stream (train episodes, then
held-out episodes); two INTACT ArchB with IDENTICAL init seeds train
open-loop via learn_transition on the identical train stream — arm E
estimated precision (default), arm F uniform_precision=True; held-out
eval via predict_reward -> |rhat - r|; terminal spikes flagged
reward >= 0.5 (same flag as K3B/K3C).

Frozen gates (VOID, not reinterpreted):
  G0a held-out terminal trials == 0 for a task -> VOID that task.
  G0b fabrication-tripwire CLEAN pre-run.
  G0c determinism spot-check: seed 74301 arm E recomputed -> b_r
      identical to 1e-12, else VOID.
  G0d hash-chained receipt + verify_chain.

Usage:
    python3 exp_precision_spike.py
Receipt: flesh-pits/receipts/EXP-FP-0008.json (hash-chained).
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
sys.path.insert(0, os.path.join(FP, "flesh-pits", "experiments", "baselines"))

from env_interface import derive_seed, config_hash  # noqa: E402
from experiments import ENVS_BY_NAME, collect_transitions  # noqa: E402
from agent import ArchB  # noqa: E402
from symbolic_baseline import SymbolicBaselineAgent  # noqa: E402
from harness import write_receipt, verify_chain  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "..", "..", "receipts")

EXP_ID = "EXP-FP-0008"
SEEDS = [74301, 74302, 74303, 74304]
TASKS = {
    # task: (n_train_episodes, n_held_episodes)
    "pomaze": (20, 10),
    "delayed_reward": (30, 15),
    "changing_rule": (30, 10),
}
SPIKE_FLAG = 0.5  # terminal-spike trial: reward >= 0.5 (K3B/K3C flag)

PREREG = {
    "hypothesis": ("The estimated-precision reward head systematically "
                   "misbehaves on rare high-reward events: across tasks "
                   "with rare terminal spikes, intact ArchB with estimated "
                   "precision shows the explosion signature on "
                   "terminal-spike trials while the uniform-precision "
                   "control stays sane; on a dense-reward task the "
                   "signature is absent."),
    "null": ("No systematic estimated-vs-uniform difference on "
             "terminal-spike trials across tasks."),
    "metric": ("Per task per seed, on held-out terminal-spike trials "
               "(reward >= 0.5): explosion signature iff "
               "mean|rerr|_est_term >= 3 * mean|rerr|_unif_term AND "
               "mean|rerr|_est_term > 0.3. Task signature iff >= 3/4 seeds. "
               "SYSTEMATIC iff >= 2/3 tasks; TASK-LOCAL iff 1; ABSENT iff 0."),
    "baseline": "arch_b uniform_precision=True (pi=1), identical streams.",
    "arms": ("E: estimated precision (default); F: uniform_precision=True. "
             "Both intact (lesion_l1=False), identical init seeds."),
    "procedure": ("Symbolic reference collects train then held-out "
                  "transitions; open-loop learn_transition training; "
                  "held-out predict_reward eval."),
    "seeds": list(SEEDS),
    "conditions": {"agent": "arch_b v1 intact", "contract": "1.0.0",
                   "tasks": {t: {"train_eps": n, "held_eps": m}
                             for t, (n, m) in TASKS.items()},
                   "spike_flag": SPIKE_FLAG},
}


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def _train_eval_precision(transitions, heldout, uniform, seed, env_name):
    """Open-loop train, then held-out mean|rerr| (+ terminal split).

    K3B _train_eval_ab instrument with the arm = precision kind
    (estimated vs uniform) instead of the L1 lesion, plus reward-head
    diagnostics: b_r, max|w_r|, mean piR on terminal training trials,
    training-stream rerr variance.
    """
    cls = ENVS_BY_NAME[env_name]
    space = cls().observation_space()
    n_actions = cls().action_space()["n"]
    agent = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, lesion_l1=False,
                  uniform_precision=uniform, seed=seed, affect="none")
    train_rerrs, term_piRs = [], []
    n_spikes = 0
    for (o, a, o2, r) in transitions:
        # Diagnostic read BEFORE the update (does not change weights;
        # expert creation is idempotent and order-identical to the
        # no-read path).
        ctx = agent.context_key(o)
        rerr = r - agent.model.predict_reward(o, a, ctx)
        train_rerrs.append(rerr)
        agent.learn_transition(o, a, o2, r)
        if r >= SPIKE_FLAG:
            n_spikes += 1
            term_piRs.append(agent.model.precR.current()[0])
    rerrs, rerrs_term = [], []
    for (o, a, o2, r) in heldout:
        ctx = agent.context_key(o)
        re_ = abs(r - agent.model.predict_reward(o, a, ctx))
        rerrs.append(re_)
        if r >= SPIKE_FLAG:
            rerrs_term.append(re_)
    var = (statistics.pvariance(train_rerrs) if len(train_rerrs) > 1
           else 0.0)
    return {
        "mean_abs_rerr": _mean(rerrs),
        "mean_abs_rerr_terminal": _mean(rerrs_term),
        "n_terminal": len(rerrs_term),
        "n_heldout": len(heldout),
        "n_train": len(transitions),
        "spike_rate_train": n_spikes / max(1, len(transitions)),
        "b_r": agent.model.b_r,
        "max_abs_w_r": max((abs(w) for w in agent.model.w_r), default=0.0),
        "mean_piR_terminal_train": _mean(term_piRs) if term_piRs else None,
        "train_rerr_var": var,
    }


def run_task(task, n_train_eps, n_held_eps):
    per_seed = []
    for si, s in enumerate(SEEDS):
        ref = SymbolicBaselineAgent()
        # Train and held-out streams collected as separate calls with
        # disjoint primary seeds -> clean split by construction.
        train_t = collect_transitions(task, n_train_eps, s, ref)
        held_t = collect_transitions(task, n_held_eps, 10_000_000 + s, ref)
        agent_seed = derive_seed(s, 0, "agent")
        arm_e = _train_eval_precision(train_t, held_t, False, agent_seed,
                                      task)
        arm_f = _train_eval_precision(train_t, held_t, True, agent_seed,
                                      task)
        est = arm_e["mean_abs_rerr_terminal"]
        unif = arm_f["mean_abs_rerr_terminal"]
        signature = (arm_e["n_terminal"] > 0 and unif > 0
                     and est >= 3.0 * unif and est > 0.3)
        per_seed.append({
            "seed": s,
            "E": {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in arm_e.items()},
            "F": {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in arm_f.items()},
            "ratio_term": round(est / unif, 4) if unif > 0 else None,
            "signature": bool(signature),
        })
        print(f"  seed {s}: E_term={est:.4f} F_term={unif:.4f} "
              f"ratio={(est / unif if unif > 0 else float('nan')):.2f} "
              f"b_r E={arm_e['b_r']:+.4f}/F={arm_f['b_r']:+.4f} "
              f"max|w_r| E={arm_e['max_abs_w_r']:.4f}/F={arm_f['max_abs_w_r']:.4f} "
              f"sig={signature}", flush=True)
    return per_seed


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tasks_out = {}
    void_tasks = []
    for task, (n_tr, n_he) in TASKS.items():
        print(f"task {task}: train {n_tr} eps / held {n_he} eps", flush=True)
        per_seed = run_task(task, n_tr, n_he)
        # G0a: held-out terminal trials == 0 -> VOID the task.
        if all(p["E"]["n_terminal"] == 0 for p in per_seed):
            print(f"  G0a: no held-out terminal trials - VOID task {task}")
            void_tasks.append(task)
            continue
        n_sig = sum(1 for p in per_seed if p["signature"])
        tasks_out[task] = {"per_seed": per_seed,
                           "task_signature": n_sig >= 3,
                           "seeds_with_signature": n_sig}

    # G0c determinism spot-check: seed 74301 pomaze arm E recomputed.
    if "pomaze" in tasks_out:
        ref = SymbolicBaselineAgent()
        train_t = collect_transitions("pomaze", TASKS["pomaze"][0],
                                      SEEDS[0], ref)
        held_t = collect_transitions("pomaze", TASKS["pomaze"][1],
                                     10_000_000 + SEEDS[0], ref)
        e_re = _train_eval_precision(
            train_t, held_t, False, derive_seed(SEEDS[0], 0, "agent"),
            "pomaze")
        e_orig = next(p for p in tasks_out["pomaze"]["per_seed"]
                      if p["seed"] == SEEDS[0])["E"]
        if abs(e_re["b_r"] - e_orig["b_r"]) > 1e-12:
            print("G0c FAIL: determinism spot-check mismatch. VOID.")
            return 2
        print(f"G0c determinism spot-check PASS "
              f"(b_r={e_re['b_r']:.6f})")

    n_task_sig = sum(1 for t in tasks_out.values() if t["task_signature"])
    if n_task_sig >= 2:
        verdict = "SYSTEMATIC"
        interp = (f"Explosion signature SYSTEMATIC: {n_task_sig}/3 tasks "
                  f"show it on >=3/4 seeds. The estimated-precision reward "
                  f"head misbehaves on rare high-reward events across "
                  f"tasks; uniform precision stays sane. Second independent "
                  f"confirmation of the C2 precision pathology family "
                  f"(pilot was lesioned/delayed_multistep; this is "
                  f"intact/multiple tasks).")
    elif n_task_sig == 1:
        verdict = "TASK-LOCAL"
        interp = (f"Explosion signature TASK-LOCAL: only "
                  f"{[t for t in tasks_out if tasks_out[t]['task_signature']]} "
                  f"shows it on >=3/4 seeds. The pathology is task-local, "
                  f"not systematic.")
    else:
        verdict = "ABSENT"
        interp = ("Explosion signature ABSENT on intact models across all "
                  "tasks: no systematic estimated-vs-uniform difference on "
                  "terminal-spike trials. The pilot sighting likely needs "
                  "the lesion's missing R_ctx absorber (or the "
                  "delayed_multistep reward structure) - that boundary is "
                  "the finding. Recorded as a characterization bound on the "
                  "C2 pathology family.")

    result = {
        "experiment_id": EXP_ID,
        "config": PREREG["conditions"],
        "config_hash": config_hash(PREREG["conditions"]),
        "primary_seed": SEEDS,
        "started_utc": started,
        "episodes": [],
        "summary": {
            "tasks": tasks_out,
            "void_tasks": void_tasks,
            "n_task_signatures": n_task_sig,
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
        limitations=("Open-loop training isolates the weight/precision "
                     "machinery (K3/K3B/K3C precedent); closed-loop control "
                     "effects factored out by design. Intact models only - "
                     "the pilot sighting was lesioned. Symbolic reference "
                     "baselines generate the transition streams "
                     "(deterministic)."),
    )
    ok, problems = verify_chain(RECEIPTS_DIR)
    print(f"hash chain: {'OK' if ok else problems}")
    print(f"\nverdict: {verdict}\n  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
