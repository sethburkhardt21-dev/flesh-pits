"""EXP-AB-K3E — branch-channel load-bearing test (delayed_reward).

Background: EXP-AB-K3D (2026-10-07) characterized the hierarchy's surviving
|e0| channel: on delayed_reward the L1-lesion gap is D_e0=+0.0986 seed-mean
(4/4 positive, 95% CI excludes 0) — the strongest hierarchy signal in the
program — and it lives in the branch channel (+0.0504, 4/4): the 3 L1
contexts (one per branch value) memorize per-branch constants that L0's
global map compromises on. Everywhere else the |e0| channel is thin or dead.

This experiment asks whether that per-branch prediction advantage is
LOAD-BEARING for control, with two preregistered arms:

  Arm A (closed-loop lesion, total causal effect): intact vs L1-lesioned
      ArchB, closed-loop on delayed_reward, learning on.
  Arm B (mechanistic planner, planning isolated from learning): one intact
      model trained open-loop (K3D-faithful); two frozen controllers from
      the same snapshot differ ONLY in the selector's vhat query —
      per-branch predict_reward (P) vs L0 global map predict_reward_global
      (G, plan_global_map=True).

PREREGISTERED (2026-10-07, before run; full JSON:
flesh-pits/experiments/preregistration_BRANCH_LOAD.json)
- carrying metric: per-seed mean episode return gap; seed-mean over 4 fresh
  seeds {77101..77104}; LOAD-BEARING fires iff seed-mean > +0.05 AND gap > 0
  on >= 3/4 seeds (both arms).
- honest prior: NULL on both arms (mechanism note in the preregistration).
- frozen gates: G0a tripwire CLEAN (done pre-run); G0b determinism
  spot-check (seed 77101 arm-A intact recompute to 1e-12); G0c hash-chained
  receipts + verify_chain; G0d crash -> VOID seed, no reseeding.

Usage:
    python3 experiments_k3e.py
"""

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
FP_EXP = os.path.join(HERE, "..", "..", "..", "flesh-pits", "experiments")
sys.path.insert(0, FP_EXP)

from env_interface import (  # noqa: E402
    config_hash, derive_seed,
)
from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, run_closed_loop, collect_transitions,
    write_receipt, _mean,
)
from agent import ArchB  # noqa: E402
from experiments_k3d import _train_agent  # noqa: E402

RECEIPTS_DIR = os.path.join(HERE, "receipts")
LANE_RECEIPTS_DIR = os.path.abspath(os.path.join(FP_EXP, "..", "receipts"))

SEEDS = [77101, 77102, 77103, 77104]
AGENT_SEED_BASE = 77120   # train/eval agent init seed base (paired arms)
REF_SEED_BASE = 77210     # reference agent seed base (arm B training stream)
N_EPS = 30                # closed-loop episodes per arm per seed


def _space():
    cls = ENVS_BY_NAME["delayed_reward"]
    return cls().observation_space(), cls().action_space()["n"]


def run_arm_a(seed, si):
    """Closed-loop lesion arm. Returns dict with per-episode returns and
    end-of-run L1 diagnostics."""
    space, n_actions = _space()
    intact = ArchB(observation_space=space, n_actions=n_actions,
                   env_name="delayed_reward", lesion_l1=False,
                   action_mode="active_inference", affect="none",
                   seed=AGENT_SEED_BASE + si)
    les = ArchB(observation_space=space, n_actions=n_actions,
                env_name="delayed_reward", lesion_l1=True,
                action_mode="active_inference", affect="none",
                seed=AGENT_SEED_BASE + si)
    ep_i = run_closed_loop("delayed_reward", intact, N_EPS, seed)
    ep_l = run_closed_loop("delayed_reward", les, N_EPS, seed)
    ri = [e["return"] for e in ep_i]
    rl = [e["return"] for e in ep_l]
    return {
        "seed": seed,
        "returns_intact": ri,
        "returns_lesioned": rl,
        "mean_intact": _mean(ri),
        "mean_lesioned": _mean(rl),
        "G_A": _mean(ri) - _mean(rl),
        "n_contexts_intact": intact.model.n_contexts,
        "n_contexts_lesioned": les.model.n_contexts,
        "D_ctx_norms_intact": {
            ",".join(map(str, k)): math.sqrt(
                sum(x * x for row in D for x in row))
            for k, D in intact.model.ctx_D.items()
        },
    }


def _frozen_controller(trained, plan_global_map, seed):
    """Frozen closed-loop controller from a trained snapshot.

    Model weights + precision state + episodic memory are restored
    identically for both planners; etas are kept at 0 (frozen); the ONLY
    difference is the selector's vhat query (plan_global_map)."""
    space, n_actions = _space()
    c = ArchB(observation_space=space, n_actions=n_actions,
              env_name="delayed_reward", action_mode="active_inference",
              affect="none", frozen=True, seed=seed)
    c.restore(trained.snapshot())
    # restore() brings back the trained (nonzero) etas — re-freeze.
    c.model.eta0 = c.model.etaD = 0.0
    c.model.eta_r = c.model.etaR = 0.0
    c.base_etas = (0.0, 0.0, 0.0, 0.0)
    c.plan_global_map = bool(plan_global_map)
    return c


def run_arm_b(seed, si):
    """Mechanistic planner arm. Trains ONE intact model open-loop
    (K3D-faithful), then races two frozen planners on the same snapshot."""
    ref = make_agent(ENVS_BY_NAME["delayed_reward"],
                     seed=REF_SEED_BASE + si, affect="none")
    stream = collect_transitions("delayed_reward", 60, seed, ref)
    train = stream[:600]
    trained = _train_agent(train, False, AGENT_SEED_BASE + si,
                           "delayed_reward")
    r_ctx = {",".join(map(str, k)): list(v)
             for k, v in trained.model.ctx_R.items()}
    p = _frozen_controller(trained, False, AGENT_SEED_BASE + si)
    g = _frozen_controller(trained, True, AGENT_SEED_BASE + si)
    ep_p = run_closed_loop("delayed_reward", p, N_EPS, seed)
    ep_g = run_closed_loop("delayed_reward", g, N_EPS, seed)
    rp = [e["return"] for e in ep_p]
    rg = [e["return"] for e in ep_g]
    return {
        "seed": seed,
        "n_train": len(train),
        "n_contexts_trained": trained.model.n_contexts,
        "R_ctx_trained": r_ctx,
        "returns_P": rp,
        "returns_G": rg,
        "mean_P": _mean(rp),
        "mean_G": _mean(rg),
        "G_B": _mean(rp) - _mean(rg),
    }


def g0b_determinism_check():
    """G0b: recompute seed 77101 arm-A intact through the identical helper;
    per-episode returns must match to 1e-12."""
    r1 = run_arm_a(SEEDS[0], 0)
    r2 = run_arm_a(SEEDS[0], 0)
    a, b = r1["returns_intact"], r2["returns_intact"]
    return (len(a) == len(b)
            and all(abs(x - y) <= 1e-12 for x, y in zip(a, b)))


K3E_PREREG = {
    "experiment_id": "EXP-AB-K3E",
    "task": "Branch-channel load-bearing test (delayed_reward).",
    "hypothesis": "The K3D branch-channel |e0| advantage translates into "
                  "closed-loop control advantage: (A) intact beats "
                  "L1-lesioned closed-loop; (B) per-branch planner beats "
                  "global-map planner on a frozen trained model.",
    "null": "G_A ~= 0 and G_B ~= 0 — prediction-only decoration.",
    "metric": "Per-seed mean episode return gap; seed-mean over 4 fresh "
              "seeds; LOAD-BEARING iff seed-mean > +0.05 AND positive on "
              ">= 3/4 seeds (both arms).",
    "baseline": "Arm A: lesion_l1=True closed-loop. Arm B: "
                "plan_global_map=True frozen planner.",
    "ablation": "L1 top-down path (arm A); prediction-only planning "
                "lesion (arm B).",
    "procedure": "Full: flesh-pits/experiments/preregistration_BRANCH_LOAD.json.",
    "seeds": list(SEEDS),
    "conditions": {
        "agent": "arch_b v1", "affect": "none",
        "action_mode": "active_inference",
        "env": "delayed_reward v1.0.0, 30 closed-loop episodes/arm/seed",
        "agent_init_base": "77120+si (paired)",
        "arm_B_ref_base": "77210+si; 60-episode ref stream, train first 600",
    },
    "decision_rule": "LOAD-BEARING iff seed-mean gap > +0.05 AND positive "
                     ">= 3/4 seeds. (A) fires -> load-bearing; (A) null (B) "
                     "fires -> partial; both null -> prediction-only "
                     "decoration (NR-B-011).",
}


def main():
    prereg = copy.deepcopy(K3E_PREREG)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    per_seed = []
    for si, s in enumerate(SEEDS):
        print(f"--- seed {s} ({si + 1}/{len(SEEDS)}) ---", flush=True)
        try:
            a = run_arm_a(s, si)
            print(f"  arm A: intact={a['mean_intact']:+.4f} "
                  f"lesioned={a['mean_lesioned']:+.4f} "
                  f"G_A={a['G_A']:+.4f} "
                  f"ctx={a['n_contexts_intact']}", flush=True)
        except Exception as e:  # noqa: BLE001 — G0d: VOID the seed
            print(f"  arm A VOID: {e!r}\n{traceback.format_exc()}",
                  flush=True)
            per_seed.append({"seed": s, "void": True,
                             "reason": f"arm A crash: {e!r}"})
            continue
        try:
            b = run_arm_b(s, si)
            print(f"  arm B: P={b['mean_P']:+.4f} G={b['mean_G']:+.4f} "
                  f"G_B={b['G_B']:+.4f} "
                  f"R_ctx={b['R_ctx_trained']}", flush=True)
        except Exception as e:  # noqa: BLE001 — G0d: VOID the seed
            print(f"  arm B VOID: {e!r}\n{traceback.format_exc()}",
                  flush=True)
            per_seed.append({"seed": s, "void": True,
                             "reason": f"arm B crash: {e!r}"})
            continue
        per_seed.append({"seed": s, "void": False, "arm_A": a, "arm_B": b})

    g0b = g0b_determinism_check()
    print(f"G0b determinism check: {'PASS' if g0b else 'FAIL'}", flush=True)

    valid = [p for p in per_seed if not p.get("void")]
    ga = [p["arm_A"]["G_A"] for p in valid]
    gb = [p["arm_B"]["G_B"] for p in valid]
    n = len(valid)
    need = 3 if n == 4 else n  # G0d: voided seed -> unanimous on remainder
    sum_a = {
        "n_valid": n,
        "per_seed_G_A": ga,
        "seed_mean_G_A": _mean(ga) if ga else 0.0,
        "n_positive_A": sum(1 for x in ga if x > 0),
        "fires_A": bool(n > 0 and _mean(ga) > 0.05
                        and sum(1 for x in ga if x > 0) >= need),
    }
    sum_b = {
        "per_seed_G_B": gb,
        "seed_mean_G_B": _mean(gb) if gb else 0.0,
        "n_positive_B": sum(1 for x in gb if x > 0),
        "fires_B": bool(n > 0 and _mean(gb) > 0.05
                        and sum(1 for x in gb if x > 0) >= need),
    }
    if sum_a["fires_A"]:
        verdict = ("LOAD-BEARING: the branch-channel |e0| advantage "
                   "translates into closed-loop control advantage.")
    elif sum_b["fires_B"]:
        verdict = ("PARTIAL: per-branch reward machinery is usable for "
                   "control but the online lesion compensates; the |e0| "
                   "channel itself is prediction-only decoration.")
    else:
        verdict = ("PREDICTION-ONLY DECORATION: the hierarchy's strongest "
                   "|e0| signal does not convert into closed-loop control "
                   "advantage on delayed_reward.")
    out = {
        "per_seed": per_seed,
        "summary_A": sum_a,
        "summary_B": sum_b,
        "G0b_determinism": g0b,
        "verdict": verdict,
    }
    print(f"\nArm A: seed-mean G_A={sum_a['seed_mean_G_A']:+.4f}, "
          f"positive {sum_a['n_positive_A']}/{n}, "
          f"fires={sum_a['fires_A']}", flush=True)
    print(f"Arm B: seed-mean G_B={sum_b['seed_mean_G_B']:+.4f}, "
          f"positive {sum_b['n_positive_B']}/{n}, "
          f"fires={sum_b['fires_B']}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    interp = ("EXP-AB-K3E branch-channel load-bearing test (4 fresh seeds, "
              "preregistered). "
              f"Arm A (closed-loop lesion): per-seed G_A = {ga}; seed-mean "
              f"{sum_a['seed_mean_G_A']:+.4f}, positive "
              f"{sum_a['n_positive_A']}/{n} -> "
              f"{'FIRES' if sum_a['fires_A'] else 'does not fire'}. "
              f"Arm B (frozen planner): per-seed G_B = {gb}; seed-mean "
              f"{sum_b['seed_mean_G_B']:+.4f}, positive "
              f"{sum_b['n_positive_B']}/{n} -> "
              f"{'FIRES' if sum_b['fires_B'] else 'does not fire'}. "
              + verdict)
    lim = ("delayed_reward only (the env where the K3D signal lives). "
           "Closed-loop ArchB (active_inference, affect='none') dithers on "
           "delayed_reward — neither arm reaches the terminal +1.0 in 30 "
           "episodes (returns are shaping-only); the return gap measures "
           "control in the dither regime. Arm B planners are frozen; "
           "learning effects factored out by design. ArchB's selector never "
           "queries predict_next (verified in agent.py _select_action) — "
           "the |e0| channel is disconnected from action selection by "
           "architecture. G0b determinism: %s."
           % ("PASS" if g0b else "FAIL"))

    # -- arch-b detail receipt (K3-lineage chain link to EXP-AB-K3D) --
    arch_path = write_receipt("EXP-AB-K3E", prereg, out, interp, lim)
    with open(arch_path) as f:
        body = json.load(f)
    k3d_path = os.path.join(RECEIPTS_DIR, "EXP-AB-K3D.json")
    with open(k3d_path, "rb") as f:
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
        "experiment_id": "EXP-AB-K3E",
        "config": {"env": "delayed_reward v1.0.0",
                   "seeds": SEEDS,
                   "arms": ["A: intact vs lesion_l1 closed-loop",
                            "B: per-branch vs global-map frozen planner"],
                   "n_episodes": N_EPS,
                   "instrument": "run_closed_loop verbatim (A); K3D-faithful "
                                 "open-loop train + frozen planners (B)"},
        "config_hash": config_hash(prereg["conditions"]),
        "primary_seed": "77101-77104",
        "started_utc": started,
        "episodes": {"n_seeds": n,
                     "voided": [p["seed"] for p in per_seed
                                if p.get("void")]},
        "summary": {
            "verdict": verdict,
            "arm_A": sum_a,
            "arm_B": sum_b,
            "G0b_determinism": g0b,
            "detail_receipt": ("prototypes/architecture-b/receipts/"
                               "EXP-AB-K3E.json"),
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
