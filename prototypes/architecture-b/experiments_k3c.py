"""EXP-AB-K3C — hierarchy beyond the two envs (B gap #3; §9 item 7).

K3B showed the L1 hierarchy earns its keep on changing_rule (reward-channel
lesion gap +0.0602) and delayed_reward (|e0| gap +0.0437). The question:
does the hierarchy earn its keep on HARDER tasks, or is it a two-env
phenomenon?

PREREGISTERED (2026-10-07, before run; full JSON:
flesh-pits/experiments/preregistration_EXP-AB-K3C.json)
- hypothesis: B's hierarchy (L1 context-expert top-down path) earns its
  keep beyond the two K3B envs: (task 1) on a longer multi-stage delayed
  task the intact-vs-L1-lesioned |e0| gap stays positive; (task 2) on a
  compositional XOR contingency the intact-vs-lesioned |rerr| gap stays
  positive (context-indexed reward tables capture the 3-way interaction
  L0's linear heads cannot represent).
- null: lesion gaps vanish (<= 0) on the harder tasks — the hierarchy
  effect is a two-env phenomenon.
- metric: per task, lesion gap D = lesioned - intact on held-out
  transitions. Task 1 (delayed_multistep) carrying: D_e0. Task 2
  (compositional_rule) carrying: D_rerr.
- baseline: arch_b lesion_l1=True (L0-only), identical transitions.
- ablation: L1 top-down path (lesion_l1) — same lesion as K3/K3B.
- procedure: open-loop transition training via learn_transition (no
  retrieval correction/affect — isolates the weight/precision machinery,
  as in K3/K3B). Intact and lesioned arms share identical transition
  streams and identical init seeds within each seed. Terminal ticks
  flagged by reward >= 0.5 (same flag as K3B).
- seeds: [74101, 74102, 74103, 74104] — fresh, no overlap with
  battery/Phase-4/repro/CALIB/consolidation seeds.
- decision_rule: Task 1 PASSES iff seed-mean D_e0 > +0.0032 AND D_e0 > 0 on
  >= 3/4 seeds. Terminal |rerr| gap is preregistered ~= 0 by construction
  (hidden 2-bit pattern) — it bounds the reward instrument, not the
  verdict. Task 2 PASSES iff seed-mean D_rerr > +0.0032 AND D_rerr > 0 on
  >= 3/4 seeds. Both PASS -> SCOPE HOLDS; one PASSES -> SCOPE BOUNDED;
  neither -> NO EFFECT (two-env phenomenon).
- conditions: contract v1.0.0; delayed_multistep v1.0.0 (30 episodes,
  train eps 0-19, held-out eps 20-29, scripted reference policy —
  ArchB/random references stall per pilot); compositional_rule v1.0.0
  (45 episodes, train eps 0-39 = 1600 transitions, held-out eps 40-44 =
  200 transitions, ArchB closed-loop reference, K3B-arm-A-faithful);
  arch_b v1, affect='none'.
- gates (VOID, not reinterpreted): G0a held-out transition count == 0
  for a task -> VOID that task.

Usage:
    python3 experiments_k3c.py
"""

import copy
import datetime
import hashlib
import json
import os
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
from experiments_phase4 import _train_eval_ab  # noqa: E402  (K3B instrument)

RECEIPTS_DIR = os.path.join(HERE, "receipts")
LANE_RECEIPTS_DIR = os.path.abspath(os.path.join(FP_EXP, "..", "receipts"))

SEEDS = [74101, 74102, 74103, 74104]
REF_SEED_BASE = 74210   # task-2 ArchB reference agent seed base
AGENT_SEED_BASE = 74220  # train/eval agent init seed base (paired arms)


def collect_scripted_multistep(n_episodes, primary_seed):
    """Per-episode transition collection with the preregistered scripted
    reference policy (deterministic from the episode seed)."""
    cls = ENVS_BY_NAME["delayed_multistep"]
    space = cls().observation_space()
    episodes = []
    for ri in range(n_episodes):
        es = derive_seed(primary_seed, ri, "env")
        branch_choice = es % 2
        sub_choice = (es // 2) % 2
        env = cls()
        obs = env.reset(es)
        trans = []
        done = False
        while not done:
            if obs["stage"] == 0 and obs["branch"] == 2:
                a = branch_choice
            elif obs["stage"] == 1 and obs["sub"] == 2:
                a = sub_choice
            else:
                a = 2  # forward
            obs2, r, done, info = env.step(a)
            trans.append((obs_to_vector(obs, space), a,
                          obs_to_vector(obs2, space), float(r)))
            obs = obs2
        episodes.append(trans)
    return episodes


def run_task1(seed, si):
    """delayed_multistep: 30 episodes, train eps 0-19, held-out eps 20-29."""
    eps = collect_scripted_multistep(30, seed)
    train = [t for ep in eps[:20] for t in ep]
    heldout = [t for ep in eps[20:] for t in ep]
    if not heldout:
        return {"void": True, "reason": "G0a: empty held-out"}
    agent_seed = AGENT_SEED_BASE + si
    intact = _train_eval_ab(train, heldout, False, agent_seed,
                            "delayed_multistep")
    les = _train_eval_ab(train, heldout, True, agent_seed,
                         "delayed_multistep")
    return {
        "seed": seed, "n_train": len(train), "n_heldout": len(heldout),
        "n_episodes": len(eps),
        "ep_lengths": [len(ep) for ep in eps],
        "intact": intact, "lesioned": les,
        "delta_e0": les["mean_abs_e0"] - intact["mean_abs_e0"],
        "delta_rerr": les["mean_abs_rerr"] - intact["mean_abs_rerr"],
        "delta_rerr_terminal": (les["mean_abs_rerr_terminal"]
                                - intact["mean_abs_rerr_terminal"]),
    }


def run_task2(seed, si):
    """compositional_rule: 45 episodes, train eps 0-39 (1600), held-out
    eps 40-44 (200). Fixed 40 steps/episode so flat slicing is
    episode-aligned (K3B-arm-A-faithful)."""
    ref = make_agent(ENVS_BY_NAME["compositional_rule"],
                     seed=REF_SEED_BASE + si, affect="none")
    all_t = collect_transitions("compositional_rule", 45, seed, ref)
    train, heldout = all_t[:1600], all_t[1600:1800]
    if not heldout:
        return {"void": True, "reason": "G0a: empty held-out"}
    agent_seed = AGENT_SEED_BASE + si
    intact = _train_eval_ab(train, heldout, False, agent_seed,
                            "compositional_rule")
    les = _train_eval_ab(train, heldout, True, agent_seed,
                         "compositional_rule")
    return {
        "seed": seed, "n_train": len(train), "n_heldout": len(heldout),
        "intact": intact, "lesioned": les,
        "delta_e0": les["mean_abs_e0"] - intact["mean_abs_e0"],
        "delta_rerr": les["mean_abs_rerr"] - intact["mean_abs_rerr"],
        "delta_rerr_terminal": (les["mean_abs_rerr_terminal"]
                                - intact["mean_abs_rerr_terminal"]),
    }


K3C_PREREG = {
    "hypothesis": "B's hierarchy (L1 context-expert top-down path) earns "
                  "its keep beyond the two K3B envs: (task 1) on a longer "
                  "multi-stage delayed task the intact-vs-L1-lesioned |e0| "
                  "gap stays positive; (task 2) on a compositional XOR "
                  "contingency the intact-vs-lesioned |rerr| gap stays "
                  "positive.",
    "null": "Lesion gaps vanish (<= 0) on the harder tasks — the hierarchy "
            "effect is a two-env phenomenon.",
    "metric": "Per task: lesion gap D = lesioned - intact on held-out "
              "transitions. Task 1 (delayed_multistep) carrying: D_e0. "
              "Task 2 (compositional_rule) carrying: D_rerr.",
    "baseline": "arch_b lesion_l1=True (L0-only), identical transitions.",
    "ablation": "L1 top-down path (lesion_l1) — same lesion as K3/K3B.",
    "procedure": "Open-loop transition training via learn_transition (no "
                 "retrieval correction/affect). Intact/lesioned arms share "
                 "identical transition streams and init seeds within each "
                 "seed. Terminal flag: reward >= 0.5. Full preregistration: "
                 "flesh-pits/experiments/preregistration_EXP-AB-K3C.json.",
    "seeds": list(SEEDS),
    "conditions": {
        "agent": "arch_b v1", "affect": "none", "contract": "1.0.0",
        "task1": "delayed_multistep v1.0.0, 30 episodes, train eps 0-19, "
                 "held-out eps 20-29, scripted reference policy",
        "task2": "compositional_rule v1.0.0, 45 episodes, train eps 0-39 "
                 "(1600 transitions), held-out eps 40-44 (200), ArchB "
                 "closed-loop reference",
    },
    "decision_rule": "Task 1 PASSES iff seed-mean D_e0 > +0.0032 AND D_e0 "
                     "> 0 on >= 3/4 seeds (terminal |rerr| preregistered "
                     "~= 0 by construction — bounds the reward instrument). "
                     "Task 2 PASSES iff seed-mean D_rerr > +0.0032 AND "
                     "D_rerr > 0 on >= 3/4 seeds. Both PASS -> SCOPE HOLDS; "
                     "one PASSES -> SCOPE BOUNDED; neither -> NO EFFECT.",
}


def _passes(per_seed, key):
    deltas = [p[key] for p in per_seed]
    return (_mean(deltas) > 0.0032 + 1e-9
            and sum(1 for d in deltas if d > 0) >= 3)


def main():
    prereg = copy.deepcopy(K3C_PREREG)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = {"task1_delayed_multistep": {"per_seed": []},
           "task2_compositional_rule": {"per_seed": []}}
    for si, s in enumerate(SEEDS):
        print(f"--- seed {s} ({si + 1}/{len(SEEDS)}) ---", flush=True)
        t1 = run_task1(s, si)
        out["task1_delayed_multistep"]["per_seed"].append(t1)
        print(f"  task1: n_train={t1.get('n_train')} "
              f"n_heldout={t1.get('n_heldout')} "
              f"d_e0={t1.get('delta_e0', float('nan')):+.4f} "
              f"d_rerr={t1.get('delta_rerr', float('nan')):+.4f} "
              f"n_term={t1.get('intact', {}).get('n_terminal')}", flush=True)
        t2 = run_task2(s, si)
        out["task2_compositional_rule"]["per_seed"].append(t2)
        print(f"  task2: n_train={t2.get('n_train')} "
              f"n_heldout={t2.get('n_heldout')} "
              f"d_e0={t2.get('delta_e0', float('nan')):+.4f} "
              f"d_rerr={t2.get('delta_rerr', float('nan')):+.4f}",
              flush=True)

    void1 = any(p.get("void") for p in out["task1_delayed_multistep"]["per_seed"])
    void2 = any(p.get("void") for p in out["task2_compositional_rule"]["per_seed"])
    t1 = out["task1_delayed_multistep"]
    t2 = out["task2_compositional_rule"]
    t1["mean_delta_e0"] = _mean([p["delta_e0"] for p in t1["per_seed"]])
    t1["mean_delta_rerr"] = _mean([p["delta_rerr"] for p in t1["per_seed"]])
    t1["mean_delta_rerr_terminal"] = _mean(
        [p["delta_rerr_terminal"] for p in t1["per_seed"]])
    t2["mean_delta_e0"] = _mean([p["delta_e0"] for p in t2["per_seed"]])
    t2["mean_delta_rerr"] = _mean([p["delta_rerr"] for p in t2["per_seed"]])
    t2["mean_delta_rerr_terminal"] = _mean(
        [p["delta_rerr_terminal"] for p in t2["per_seed"]])

    pass1 = (not void1) and _passes(t1["per_seed"], "delta_e0")
    pass2 = (not void2) and _passes(t2["per_seed"], "delta_rerr")
    t1["passes_gate"] = pass1
    t2["passes_gate"] = pass2

    if void1 or void2:
        verdict = "VOID"
        interp = ("VOID per G0a: empty held-out on task1=%s task2=%s. "
                  "No interpretation." % (void1, void2))
    elif pass1 and pass2:
        verdict = "SCOPE HOLDS"
        interp = ("SCOPE HOLDS: the hierarchy earns its keep beyond the two "
                  "K3B envs — task1 (delayed_multistep) D_e0 seed-mean "
                  f"{t1['mean_delta_e0']:+.4f}, task2 (compositional_rule) "
                  f"D_rerr seed-mean {t2['mean_delta_rerr']:+.4f}, both "
                  "gates passed.")
    elif pass1 or pass2:
        held = "task1 (delayed_multistep, longer multi-stage horizon)" \
            if pass1 else "task2 (compositional_rule, XOR contingency)"
        failed = "task2 (compositional_rule)" \
            if pass1 else "task1 (delayed_multistep)"
        verdict = "SCOPE BOUNDED"
        interp = (f"SCOPE BOUNDED: the hierarchy earns its keep on {held} "
                  f"(gate passed) but not on {failed} (gate failed) — "
                  f"task1 D_e0 seed-mean {t1['mean_delta_e0']:+.4f}, task2 "
                  f"D_rerr seed-mean {t2['mean_delta_rerr']:+.4f}. The "
                  "hierarchy's scope is bounded; record the task "
                  "characteristics where it holds vs fails.")
    else:
        verdict = "NO EFFECT"
        interp = ("NO EFFECT on harder tasks: lesion gaps vanish on both "
                  f"new tasks (task1 D_e0 seed-mean "
                  f"{t1['mean_delta_e0']:+.4f}, task2 D_rerr seed-mean "
                  f"{t2['mean_delta_rerr']:+.4f}). The hierarchy effect is "
                  "a two-env phenomenon — K3B's gaps do not generalize.")
    out["verdict"] = verdict
    lim = ("Open-loop training isolates the weight/precision machinery (as "
           "K3/K3B); closed-loop control effects are factored out by "
           "design. Task-1 reference is a scripted policy (ArchB/random "
           "references stall per pilot); task-2 reference is ArchB "
           "closed-loop (K3B-arm-A-faithful). Terminal ticks flagged by "
           "reward >= 0.5. 4 fresh seeds.")

    # -- arch-b receipt (K3B schema + hash-chain link to EXP-AB-K3B) --
    arch_path = write_receipt("EXP-AB-K3C", prereg, out, interp, lim)
    with open(arch_path) as f:
        body = json.load(f)
    k3b_path = os.path.join(RECEIPTS_DIR, "EXP-AB-K3B.json")
    with open(k3b_path, "rb") as f:
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
        "experiment_id": "EXP-AB-K3C",
        "config": {"tasks": ["delayed_multistep v1.0.0",
                             "compositional_rule v1.0.0"],
                   "seeds": SEEDS, "lesion": "lesion_l1",
                   "instrument": "K3B open-loop (learn_transition / "
                                 "eval_transition / predict_reward)"},
        "config_hash": config_hash(prereg["conditions"]),
        "primary_seed": "74101-74104",
        "started_utc": started,
        "episodes": {
            "delayed_multistep": {
                "n_train": t1["per_seed"][0].get("n_train"),
                "n_heldout": t1["per_seed"][0].get("n_heldout")},
            "compositional_rule": {
                "n_train": t2["per_seed"][0].get("n_train"),
                "n_heldout": t2["per_seed"][0].get("n_heldout")},
        },
        "summary": {
            "verdict": verdict,
            "task1_delayed_multistep": {
                "seed_mean_delta_e0": t1["mean_delta_e0"],
                "seed_mean_delta_rerr": t1["mean_delta_rerr"],
                "seed_mean_delta_rerr_terminal":
                    t1["mean_delta_rerr_terminal"],
                "passes_gate": pass1,
                "per_seed_delta_e0": [p["delta_e0"]
                                      for p in t1["per_seed"]]},
            "task2_compositional_rule": {
                "seed_mean_delta_e0": t2["mean_delta_e0"],
                "seed_mean_delta_rerr": t2["mean_delta_rerr"],
                "passes_gate": pass2,
                "per_seed_delta_rerr": [p["delta_rerr"]
                                        for p in t2["per_seed"]]},
            "detail_receipt": ("prototypes/architecture-b/receipts/"
                               "EXP-AB-K3C.json"),
        },
    }
    lane_path = lane_write_receipt(
        lane_result, LANE_RECEIPTS_DIR,
        hypothesis=prereg["hypothesis"], null=prereg["null"],
        preregistered_metric=prereg["metric"], baseline=prereg["baseline"],
        conditions=json.dumps(prereg["conditions"], sort_keys=True),
        interpretation=interp, limitations=lim)
    print(f"\nVERDICT: {verdict}\n  arch receipt: {arch_path}\n  lane receipt: {lane_path}")


if __name__ == "__main__":
    main()
