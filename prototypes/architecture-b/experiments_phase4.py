"""Phase-4 gap-closure experiments — Architecture B.

M1  : no-memory ablation (pomaze, delayed_reward) — recorded gap 2.
C2B : shift-robust precision estimator vs uniform/estimated — recorded gap 5.
C4B : uncertainty-reduction information gain (probe-error metric) — gap 5.
K3B : longer-horizon hierarchy test (changing_rule x5 training; delayed_reward).

Every experiment is PREREGISTERED in code (hypothesis, null, preregistered
metric, baseline, ablation, procedure, seeds) BEFORE the run. Results go to
receipts/ as JSON with interpretation and limitations appended after the run.
Multi-seed: every experiment runs on 3 fresh seeds (300s/310s/320s/330s —
none used by the Phase-3 battery).

Conventions match experiments.py: canonical flesh-pits envs (read-only),
one env instance per run, one agent instance across episodes, deterministic
seeded RNG. CONSCIOUSNESS: UNRESOLVED — mechanisms only.

Usage:
    python3 experiments_phase4.py --exp M1
    python3 experiments_phase4.py --exp ALL
"""

import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP_EXP = os.path.join(HERE, "..", "..", "..", "flesh-pits", "experiments")
sys.path.insert(0, FP_EXP)

from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, run_closed_loop, collect_transitions,
    write_receipt, _mean,
)
from agent import ArchB  # noqa: E402


# ---------------------------------------------------------------------------
# M1 — no-memory ablation (recorded gap: EpisodicStore causal status unknown)
# ---------------------------------------------------------------------------
M1_PREREG = {
    "hypothesis": "Disabling episodic memory produces a SELECTIVE deficit: "
                  "pomaze (partial observability; recall should aid "
                  "revisit-avoidance via retrieval correction) degrades more "
                  "than delayed_reward (credit flows through the "
                  "reward-channel learning path).",
    "null": "Disabling memory changes nothing on either task — the "
            "EpisodicStore is decorative in Architecture B.",
    "metric": "mean_return per task; per-seed delta = intact − no_memory; "
              "seed-mean deltas compared across tasks.",
    "baseline": "arch_b memory_enabled=True (intact store).",
    "ablation": "arch_b memory_enabled=False: DisabledStore — no encoding, "
                "no retrieval correction, per-action error prediction "
                "returns None (selector falls back to its running table).",
    "procedure": "Closed-loop runs: pomaze 15 episodes, delayed_reward 10 "
                 "episodes; affect='none' (world-model isolation); "
                 "action_mode='active_inference'; identical primary seeds "
                 "for both conditions (paired).",
    "seeds": [301, 302, 303],
    "conditions": {"agent": "arch_b v1", "affect": "none",
                   "action_mode": "active_inference"},
    "decision_rule": "Selective deficit (pomaze seed-mean delta > 0 and "
                     "> delayed_reward seed-mean delta) -> memory earns a "
                     "causal role. Seed-mean deltas ≈ 0 on both tasks -> "
                     "memory decorative in B (NR-B-007).",
}


def exp_m1():
    prereg = copy.deepcopy(M1_PREREG)
    seeds = prereg["seeds"]
    out = {}
    for env_name, n_eps in (("pomaze", 15), ("delayed_reward", 10)):
        per_seed = []
        for s in seeds:
            a_intact = make_agent(ENVS_BY_NAME[env_name], seed=1000 + s,
                                  affect="none", memory_enabled=True)
            eps_i = run_closed_loop(env_name, a_intact, n_eps, s)
            a_nomem = make_agent(ENVS_BY_NAME[env_name], seed=1000 + s,
                                 affect="none", memory_enabled=False)
            eps_n = run_closed_loop(env_name, a_nomem, n_eps, s)
            mi = _mean([e["return"] for e in eps_i])
            mn = _mean([e["return"] for e in eps_n])
            per_seed.append({"seed": s, "intact": mi, "no_memory": mn,
                             "delta": mi - mn,
                             "intact_memories": len(a_intact.memory),
                             "nomem_memories": len(a_nomem.memory)})
        out[env_name] = {"per_seed": per_seed,
                         "mean_delta": _mean([p["delta"] for p in per_seed])}
    d_pom = out["pomaze"]["mean_delta"]
    d_drw = out["delayed_reward"]["mean_delta"]
    selective = d_pom > 0 and d_pom > d_drw + 1e-9
    decorative = abs(d_pom) < 0.05 and abs(d_drw) < 0.05
    if selective and not decorative:
        interp = (f"SELECTIVE DEFICIT as preregistered: pomaze delta "
                  f"{d_pom:+.3f} > delayed_reward delta {d_drw:+.3f}. "
                  f"Episodic memory is causally load-bearing on the "
                  f"partial-observability task.")
    elif decorative:
        interp = (f"DECORATIVE: pomaze delta {d_pom:+.3f}, delayed_reward "
                  f"delta {d_drw:+.3f} — disabling the store changed nothing "
                  f"on either task. The EpisodicStore is decorative in B.")
    else:
        interp = (f"MIXED: pomaze delta {d_pom:+.3f}, delayed_reward delta "
                  f"{d_drw:+.3f} — no clean selective deficit; memory effect "
                  f"is task-dependent and weak.")
    lim = ("DisabledStore also removes the per-action error signal used by "
           "the AI selector's IG term (falls back to the running table), so "
           "the ablation covers the store's full causal footprint. 3 seeds; "
           "affect='none' isolates the world model.")
    path = write_receipt("EXP-AB-M1", prereg, out, interp, lim)
    print(f"M1: pomaze d={d_pom:+.3f} delayed_reward d={d_drw:+.3f}\n  {path}")
    return out, decorative


# ---------------------------------------------------------------------------
# C2B — shift-robust precision estimator (recorded gap: C2 metric flaw)
# ---------------------------------------------------------------------------
C2B_PREREG = {
    "hypothesis": "ShiftRobustPrecisionEstimator (surprise-triggered window "
                  "reset, k=4.0) matches or beats estimated precision and "
                  "beats uniform pi=1 on noisy changing_rule, without losing "
                  "to uniform on a stationary noisy task.",
    "null": "Uniform >= shift-reset on both tasks — the precision idea is "
            "dead even in shift-aware form.",
    "metric": "mean_return over episodes, seed-meaned. Shift arm: 10 "
              "episodes changing_rule (1 rule flip). Stationary arm: 5 "
              "episodes changing_rule phase-0 only (rule fixed, 10% noise).",
    "baseline": "arch_b uniform pi=1; arch_b estimated pi (C2 replication).",
    "ablation": "arch_b precision_kind='shift_reset', surprise_k=4.0 "
                "(preregistered; ONE variant).",
    "procedure": "Closed-loop runs, affect='none', identical primary seeds "
                 "across the three conditions (paired). Diagnostic: window "
                 "reset counts — expected ~1 per flip on the shift arm, ~0 "
                 "on the stationary arm.",
    "seeds": [311, 312, 313],
    "conditions": {"agent": "arch_b v1", "affect": "none",
                   "surprise_k": 4.0},
    "revival_rule": "Revival iff (shift: shift_reset seed-mean >= "
                    "estimated seed-mean AND > uniform seed-mean) AND "
                    "(stationary: shift_reset seed-mean >= uniform "
                    "seed-mean). Otherwise it joins the rejected list "
                    "(NR-B-008).",
}

_C2B_MODES = {
    "uniform": {"uniform_precision": True},
    "estimated": {},
    "shift_reset": {"precision_kind": "shift_reset", "surprise_k": 4.0},
}


def _c2b_resets(agent):
    m = agent.model
    return (m.prec0.resets + m.prec1.resets + m.precR.resets
            if hasattr(m.prec0, "resets") else 0)


def exp_c2b():
    prereg = copy.deepcopy(C2B_PREREG)
    seeds = prereg["seeds"]
    out = {}
    for arm, n_eps in (("shift", 10), ("stationary", 5)):
        arm_out = {}
        for mode, kw in _C2B_MODES.items():
            per_seed, resets = [], []
            for s in seeds:
                agent = make_agent(ENVS_BY_NAME["changing_rule"],
                                   seed=1100 + s, affect="none", **kw)
                eps = run_closed_loop("changing_rule", agent, n_eps, s)
                per_seed.append(_mean([e["return"] for e in eps]))
                resets.append(_c2b_resets(agent))
            arm_out[mode] = {"per_seed_returns": per_seed,
                             "mean_return": _mean(per_seed),
                             "per_seed_resets": resets,
                             "mean_resets": _mean(resets)}
        out[arm] = arm_out
    sh, st = out["shift"], out["stationary"]
    revival = (sh["shift_reset"]["mean_return"]
               >= sh["estimated"]["mean_return"] - 1e-9
               and sh["shift_reset"]["mean_return"]
               > sh["uniform"]["mean_return"] + 1e-9
               and st["shift_reset"]["mean_return"]
               >= st["uniform"]["mean_return"] - 1e-9)
    if revival:
        interp = (f"REVIVAL WITH CAVEATS: shift_reset shift "
                  f"{sh['shift_reset']['mean_return']:.2f} >= estimated "
                  f"{sh['estimated']['mean_return']:.2f} > uniform "
                  f"{sh['uniform']['mean_return']:.2f}; stationary "
                  f"{st['shift_reset']['mean_return']:.2f} >= uniform "
                  f"{st['uniform']['mean_return']:.2f}. Precision survives "
                  f"only in the shift-aware form.")
    else:
        interp = (f"REJECTED: shift_reset shift "
                  f"{sh['shift_reset']['mean_return']:.2f} vs estimated "
                  f"{sh['estimated']['mean_return']:.2f} vs uniform "
                  f"{sh['uniform']['mean_return']:.2f}; stationary "
                  f"{st['shift_reset']['mean_return']:.2f} vs uniform "
                  f"{st['uniform']['mean_return']:.2f}. The revival rule "
                  f"failed — shift-aware precision joins the rejected list.")
    lim = ("ONE preregistered variant (k=4.0); no tuning after the run. "
           "Stationary arm is 5 phase-0 episodes (200 trials). 3 seeds. "
           "affect='none' isolates the world model.")
    path = write_receipt("EXP-AB-C2B", prereg, out, interp, lim)
    print(f"C2B: revival={revival}\n  {path}")
    return out, not revival


# ---------------------------------------------------------------------------
# C4B — uncertainty-reduction information gain (recorded gap: C4 metric flaw)
# ---------------------------------------------------------------------------
C4B_PREREG = {
    "hypothesis": "Active-inference action selection produces larger "
                  "uncertainty reduction than greedy/random, measured as "
                  "probe-set prediction-error decline (not error decline "
                  "within predictable regions).",
    "null": "AI probe-error reduction <= greedy/random — the IG term is "
            "not causal; the 'active inference' label stays off permanently.",
    "metric": "IG_probe = mean|e0| on a FIXED probe set before − after "
              "15-episode pomaze training; seed-meaned. Secondary: "
              "mean_return.",
    "baseline": "arch_b with action_mode greedy / random (same world "
                "model, same seeds — only the act() policy differs).",
    "ablation": "The action policy (active_inference vs greedy vs random).",
    "procedure": "Probe: 400 transitions from random-policy pomaze "
                 "rollouts (fixed seeds, policy-independent). Per "
                 "(seed, mode): fresh agent with IDENTICAL init across "
                 "modes -> probe-before -> 15 closed-loop episodes -> "
                 "probe-after. Probe eval is pure model prediction "
                 "(eval_transition: no memory store, no retrieval "
                 "correction) — it measures model knowledge, not memory.",
    "seeds": [321, 322, 323],
    "conditions": {"env": "pomaze v1.0.0", "episodes": 15,
                   "agent": "arch_b v1", "affect": "none",
                   "probe_seed": 901, "probe_size": 400},
    "kill_rule": "AI seed-mean IG_probe <= max(greedy, random) seed-mean "
                 "IG_probe -> label stays off PERMANENTLY (NR-B-009).",
}


def _collect_probe():
    rand_agent = make_agent(ENVS_BY_NAME["pomaze"], seed=42,
                            action_mode="random", affect="none")
    trans = collect_transitions("pomaze", 6, 901, rand_agent)
    return trans[:400]


def _probe_error(agent, probe):
    return _mean([agent.eval_transition(o, a, o2)
                  for (o, a, o2, r) in probe])


def exp_c4b():
    prereg = copy.deepcopy(C4B_PREREG)
    seeds = prereg["seeds"]
    probe = _collect_probe()
    out = {}
    for mode in ("active_inference", "greedy", "random"):
        per_seed = []
        for si, s in enumerate(seeds):
            agent = make_agent(ENVS_BY_NAME["pomaze"], seed=500 + si,
                               action_mode=mode, affect="none")
            before = _probe_error(agent, probe)
            eps = run_closed_loop("pomaze", agent, 15, s)
            after = _probe_error(agent, probe)
            per_seed.append({
                "seed": s,
                "probe_before": before, "probe_after": after,
                "IG_probe": before - after,
                "mean_return": _mean([e["return"] for e in eps]),
            })
        out[mode] = {"per_seed": per_seed,
                     "mean_IG_probe": _mean([p["IG_probe"]
                                             for p in per_seed]),
                     "mean_return": _mean([p["mean_return"]
                                           for p in per_seed])}
    ai = out["active_inference"]["mean_IG_probe"]
    gr = out["greedy"]["mean_IG_probe"]
    ra = out["random"]["mean_IG_probe"]
    survives = ai > gr + 1e-9 and ai > ra + 1e-9
    if survives:
        interp = (f"SURVIVES: AI IG_probe {ai:.4f} > greedy {gr:.4f} and > "
                  f"random {ra:.4f} on the uncertainty-reduction metric. The "
                  f"IG term is causal; the label may be reconsidered with "
                  f"caveats (returns: ai "
                  f"{out['active_inference']['mean_return']:.2f}, greedy "
                  f"{out['greedy']['mean_return']:.2f}, random "
                  f"{out['random']['mean_return']:.2f}).")
    else:
        interp = (f"KILL CONFIRMED PERMANENTLY: AI IG_probe {ai:.4f} vs "
                  f"greedy {gr:.4f} vs random {ra:.4f} — the sharper metric "
                  f"does not revive active inference. The label stays off.")
    lim = ("Probe drawn from a random policy: it matches random's "
           "experience distribution, which if anything favors random — a "
           "conservative test for AI. Identical init across modes makes "
           "probe-before identical by construction. 3 seeds; affect='none'.")
    path = write_receipt("EXP-AB-C4B", prereg, out, interp, lim)
    print(f"C4B: ai={ai:.4f} greedy={gr:.4f} random={ra:.4f} "
          f"survives={survives}\n  {path}")
    return out, not survives


# ---------------------------------------------------------------------------
# K3B — longer-horizon hierarchy test (recorded gap: K3 weak effect)
# ---------------------------------------------------------------------------
K3B_PREREG = {
    "hypothesis": "With longer training on structured tasks, the "
                  "intact-vs-L1-lesioned prediction gap GROWS beyond K3's "
                  "+0.0032: the top-down context experts earn their keep "
                  "where predictable structure spans longer horizons.",
    "null": "Gaps stay tiny/absent — the hierarchy effect is bounded at "
            "K3's weak level.",
    "metric": "Arm A (changing_rule, 1500 train transitions, held-out 200 "
              "from a stable phase): mean|e0| and mean|rerr| intact vs "
              "lesioned. Arm B (delayed_reward, 600 train / 120 held-out): "
              "same metrics; preregistered expectation: terminal |rerr| "
              "gap ≈ 0 because the correct branch is hidden (50/50 "
              "unpredictable from obs) — arm B bounds the test, arm A "
              "carries it.",
    "baseline": "arch_b lesion_l1=True (L0-only), identical transitions.",
    "ablation": "L1 top-down path (lesion_l1) — same lesion as K3.",
    "procedure": "Open-loop transition training via learn_transition "
                 "(no retrieval correction/affect — isolates the "
                 "weight/precision machinery, as in K3), 3 seeds.",
    "seeds": [331, 332, 333],
    "conditions": {"agent": "arch_b v1", "affect": "none"},
    "decision_rule": "Arm-A |e0| or |rerr| lesion gap clearly larger than "
                     "K3's +0.0032 (seed-mean) -> hierarchy earns its keep. "
                     "Gaps at K3 scale or smaller -> effect stays weak; "
                     "note as a bound.",
}


def _train_eval_ab(transitions, heldout, lesion_l1, seed, env_name):
    """Open-loop train, then held-out mean|e0| and mean|rerr|."""
    cls = ENVS_BY_NAME[env_name]
    space = cls().observation_space()
    n_actions = cls().action_space()["n"]
    agent = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, lesion_l1=lesion_l1, seed=seed,
                  affect="none")
    for (o, a, o2, r) in transitions:
        agent.learn_transition(o, a, o2, r)
    e0s, rerrs, rerrs_term = [], [], []
    for (o, a, o2, r) in heldout:
        e0s.append(agent.eval_transition(o, a, o2))
        ctx = agent.context_key(o)
        re_ = abs(r - agent.model.predict_reward(o, a, ctx))
        rerrs.append(re_)
        if r >= 0.5:  # terminal tick (delayed_reward final reward)
            rerrs_term.append(re_)
    return {"mean_abs_e0": _mean(e0s), "mean_abs_rerr": _mean(rerrs),
            "mean_abs_rerr_terminal": _mean(rerrs_term),
            "n_terminal": len(rerrs_term)}


def exp_k3b():
    prereg = copy.deepcopy(K3B_PREREG)
    seeds = prereg["seeds"]
    out = {}
    # Arm A: changing_rule, 45 episodes = 1800 transitions; train on the
    # first 1500, held-out on episodes 40-44 (stable phase 8).
    arm_a = {"per_seed": []}
    for si, s in enumerate(seeds):
        ref = make_agent(ENVS_BY_NAME["changing_rule"], seed=900 + si,
                         affect="none")
        all_t = collect_transitions("changing_rule", 45, s, ref)
        train, heldout = all_t[:1500], all_t[1600:1800]
        intact = _train_eval_ab(train, heldout, False, 910 + si,
                                "changing_rule")
        les = _train_eval_ab(train, heldout, True, 910 + si,
                             "changing_rule")
        arm_a["per_seed"].append({
            "seed": s, "n_train": len(train), "n_heldout": len(heldout),
            "intact": intact, "lesioned": les,
            "delta_e0": les["mean_abs_e0"] - intact["mean_abs_e0"],
            "delta_rerr": les["mean_abs_rerr"] - intact["mean_abs_rerr"]})
    arm_a["mean_delta_e0"] = _mean([p["delta_e0"]
                                    for p in arm_a["per_seed"]])
    arm_a["mean_delta_rerr"] = _mean([p["delta_rerr"]
                                      for p in arm_a["per_seed"]])
    out["arm_a_changing_rule"] = arm_a
    # Arm B: delayed_reward, 60 episodes; train 600, held-out last 120.
    arm_b = {"per_seed": []}
    for si, s in enumerate(seeds):
        ref = make_agent(ENVS_BY_NAME["delayed_reward"], seed=920 + si,
                         affect="none")
        all_t = collect_transitions("delayed_reward", 60, s, ref)
        train, heldout = all_t[:600], all_t[-120:]
        intact = _train_eval_ab(train, heldout, False, 930 + si,
                                "delayed_reward")
        les = _train_eval_ab(train, heldout, True, 930 + si,
                             "delayed_reward")
        arm_b["per_seed"].append({
            "seed": s, "n_train": len(train), "n_heldout": len(heldout),
            "intact": intact, "lesioned": les,
            "delta_e0": les["mean_abs_e0"] - intact["mean_abs_e0"],
            "delta_rerr": les["mean_abs_rerr"] - intact["mean_abs_rerr"],
            "delta_rerr_terminal":
                les["mean_abs_rerr_terminal"]
                - intact["mean_abs_rerr_terminal"]})
    arm_b["mean_delta_e0"] = _mean([p["delta_e0"]
                                    for p in arm_b["per_seed"]])
    arm_b["mean_delta_rerr"] = _mean([p["delta_rerr"]
                                      for p in arm_b["per_seed"]])
    arm_b["mean_delta_rerr_terminal"] = _mean(
        [p["delta_rerr_terminal"] for p in arm_b["per_seed"]])
    out["arm_b_delayed_reward"] = arm_b

    grown = (arm_a["mean_delta_e0"] > 0.0032 + 1e-9
             or arm_a["mean_delta_rerr"] > 0.0032 + 1e-9)
    if grown:
        interp = (f"HIERARCHY EARNS ITS KEEP: arm-A lesion gaps grew vs "
                  f"K3's +0.0032 — |e0| gap "
                  f"{arm_a['mean_delta_e0']:+.4f}, |rerr| gap "
                  f"{arm_a['mean_delta_rerr']:+.4f} (seed-means). Arm B: "
                  f"|e0| {arm_b['mean_delta_e0']:+.4f}, terminal |rerr| "
                  f"{arm_b['mean_delta_rerr_terminal']:+.4f} (expected "
                  f"≈0 — hidden branch bounds it).")
    else:
        interp = (f"EFFECT STAYS WEAK: arm-A lesion gaps at/below K3 scale "
                  f"— |e0| gap {arm_a['mean_delta_e0']:+.4f}, |rerr| gap "
                  f"{arm_a['mean_delta_rerr']:+.4f} (seed-means). Recorded "
                  f"as a bound on the hierarchy's contribution.")
    lim = ("Open-loop training isolates the weight/precision machinery "
           "(as K3); closed-loop control effects are factored out by "
           "design. Arm-B terminal ticks flagged by reward >= 0.5. 3 seeds.")
    path = write_receipt("EXP-AB-K3B", prereg, out, interp, lim)
    print(f"K3B: armA d_e0={arm_a['mean_delta_e0']:+.4f} "
          f"d_rerr={arm_a['mean_delta_rerr']:+.4f} grown={grown}\n  {path}")
    return out, not grown


EXPS = {
    "M1": exp_m1,
    "C2B": exp_c2b,
    "C4B": exp_c4b,
    "K3B": exp_k3b,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL",
                    choices=sorted(EXPS) + ["ALL"])
    args = ap.parse_args(argv)
    names = sorted(EXPS) if args.exp == "ALL" else [args.exp]
    summary = {}
    for name in names:
        print(f"=== EXP-AB-{name} ===")
        try:
            result, bad = EXPS[name]()
            summary[name] = {"kill_or_fail": bool(bad)}
        except Exception as ex:  # fail visible, never silent
            print(f"EXP-AB-{name} ERROR: {type(ex).__name__}: {ex}")
            summary[name] = {"error": f"{type(ex).__name__}: {ex}"}
    print("\n=== SUMMARY ===")
    for name, s in summary.items():
        print(f"EXP-AB-{name}: {s}")


if __name__ == "__main__":
    main()
