"""EXP-FP-0011-ALIASING — causal test: is obs aliasing THE bound on
arch-B's pomaze performance?

Preregistered: experiments/preregistration_ALIASING.json (sealed BEFORE run).

Follow-up to EXP-FP-0010-POMAZE-DIAG, which diagnosed B as
EXPLORATION-dominant + REPRESENTATION-bound (aliasing rate 0.667,
n_contexts=1, p_goal 0.133). This experiment removes the aliasing
ADDITIVELY — a driver-side odometry wrapper (experiments/odom_wrapper.py)
appends a coarse relative-position hash using only the obs stream +
chosen actions (no goal position, no walls, no absolute position, no env
internals) — and measures pomaze mean return + p_goal vs unmodified B on
6 fresh paired seeds.

Arms (15 episodes/arm/seed, one persistent agent per (arm, seed)):
  B-BASE    : ArchB on canonical POMaze (baseline replication)
  B-AUG     : ArchB on POMaze+odom (drive='true')
  B-SHUF    : ArchB on POMaze+odom (drive='shuffle') — §40 kill arm:
              same channel format, odometry decorrelated from position
  B-BASE-CR : ArchB on canonical ChangingRule (control baseline)
  B-AUG-CR  : ArchB on ChangingRule+odom (inert: constant channels)

Agent config byte-identical to EXP-FP-0010 make_B
(action_mode='active_inference', affect='none', default etas).

Frozen decision rules (in the preregistration):
  H supported iff p_goal_AUG >= 2*p_goal_BASE (floor: >=0.20 if base is 0)
    AND mean_return_AUG - mean_return_BASE >= 0.5 (6-seed aggregates).
  Null holds iff neither fires. Exactly one -> PARTIAL.
  Kill: H supported but B-SHUF also fires both gates -> H REJECTED
    (capacity/noise confound).
  Control: mean_return_AUG_CR - mean_return_BASE_CR >= -1.0.

Receipts: hash-chained ndjson receipts/EXP-FP-0011-ALIASING.ndjson.
Canonical env/agent files untouched (sha256 recorded, G0). Nothing pushed.

Stdlib only. Deterministic given seeds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_B = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-b")
_ENVS = os.path.join(_HERE, "envs")
for _p in (_ARCH_B, _HERE, _ENVS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import ArchB  # noqa: E402
from env_interface import derive_seed  # noqa: E402
from pomaze import POMaze  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402
from odom_wrapper import OdomWrapper  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0011-ALIASING"
SEEDS = (93001, 93002, 93003, 93004, 93005, 93006)
N_EPS = 15

ARMS = ["B-BASE", "B-AUG", "B-SHUF", "B-BASE-CR", "B-AUG-CR"]


def make_env(arm):
    """Returns (step_env, raw_env). step_env exposes the agent-facing API
    (reset/step/observation_space/action_space/NAME); raw_env is the
    underlying canonical env for driver-side analysis (env._pos etc.)."""
    if arm == "B-BASE":
        env = POMaze()
        return env, env
    if arm == "B-AUG":
        wrap = OdomWrapper(POMaze(), drive="true")
        return wrap, wrap.env
    if arm == "B-SHUF":
        wrap = OdomWrapper(POMaze(), drive="shuffle")
        return wrap, wrap.env
    if arm == "B-BASE-CR":
        env = ChangingRule()
        return env, env
    if arm == "B-AUG-CR":
        wrap = OdomWrapper(ChangingRule(), drive="true")
        return wrap, wrap.env
    raise ValueError(f"unknown arm {arm!r}")


def run_B_arm(arm, primary_seed, n_episodes):
    """ArchB closed-loop. Returns (agent, ep_summaries). Agent config
    byte-identical to EXP-FP-0010 make_B."""
    env, raw = make_env(arm)
    space = env.observation_space()
    n_actions = env.action_space()["n"]
    agent = ArchB(space, n_actions, env_name=env.NAME,
                  action_mode="active_inference", affect="none",
                  seed=derive_seed(primary_seed, 0, "agent"))
    ep_summaries = []
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        if isinstance(env, OdomWrapper):
            obs = env.reset(ep_seed, run_index)
        else:
            obs = env.reset(ep_seed)
        agent.reset(derive_seed(ep_seed, run_index, "agent"),
                    env.action_space())
        rerrs = []
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick_n = 0
        done = False
        while not done:
            pos_before = getattr(raw, "_pos", None)
            a = agent.act(obs)
            rhat_chosen = float(agent._pending["rhat"])
            obs, r, done, info = env.step(a)
            agent.update(obs, a, r, done, info)
            rerrs.append(abs(r - rhat_chosen))
            total += r
            tick_n += 1
            if pos_before is not None and getattr(raw, "_pos", None) \
                    == pos_before:
                stuck += 1
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick_n
            if done:
                break
        ep_summaries.append({
            "return": round(total, 4), "goal": goal,
            "steps_to_goal": steps_to_goal,
            "stuck_frac": (round(stuck / max(1, tick_n), 4)
                           if pos_before is not None else None),
            "ticks": tick_n,
            "mean_abs_rerr": round(sum(rerrs) / max(1, len(rerrs)), 5),
            "n_contexts": agent.model.n_contexts,
        })
    return agent, ep_summaries


# ---------------------------------------------------------------------------
# hash-chained receipt emission (K10 / pomaze_diag convention)
# ---------------------------------------------------------------------------
def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


class Emitter:
    def __init__(self):
        self.records = []
        self.t_wall = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()

    def emit(self, kind, payload):
        rec = {"experiment_id": EXPERIMENT_ID, "kind": kind,
               "t_wall": self.t_wall}
        rec.update(payload)
        rec["prev_hash"] = (self.records[-1]["receipt_hash"]
                            if self.records else "GENESIS")
        rec["receipt_hash"] = receipt_hash(rec)
        self.records.append(rec)
        return rec

    def verify(self):
        prev = "GENESIS"
        for rec in self.records:
            if rec["prev_hash"] != prev:
                return False, f"link break at {rec['kind']}"
            body = {k: v for k, v in rec.items() if k != "receipt_hash"}
            if receipt_hash(body) != rec["receipt_hash"]:
                return False, f"hash mismatch at {rec['kind']}"
            prev = rec["receipt_hash"]
        return True, f"{len(self.records)} records chained"


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    t0 = time.time()
    em = Emitter()
    lab = os.path.dirname(_HERE)

    # G0: canonical files untouched — sha256 recorded pre-run.
    g0_files = {
        "experiments/envs/pomaze.py": os.path.join(_ENVS, "pomaze.py"),
        "experiments/envs/changing_rule.py": os.path.join(
            _ENVS, "changing_rule.py"),
        "experiments/env_interface.py": os.path.join(_HERE,
                                                     "env_interface.py"),
        "prototypes/architecture-b/agent.py": os.path.join(
            _ARCH_B, "agent.py"),
        "prototypes/architecture-b/generative_model.py": os.path.join(
            _ARCH_B, "generative_model.py"),
    }
    g0_pre = {k: sha256_file(p) for k, p in g0_files.items()}
    with open(os.path.join(_HERE, "preregistration_ALIASING.json")) as f:
        prereg = json.load(f)
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_ALIASING.json",
        "prereg_sha256": hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()).hexdigest(),
        "experiment_id": EXPERIMENT_ID,
        "type": prereg["type"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "episodes_per_arm_per_seed": N_EPS,
        "g0_sha256_pre": g0_pre,
        "new_files_sha256": {
            "experiments/odom_wrapper.py": sha256_file(
                os.path.join(_HERE, "odom_wrapper.py")),
            "experiments/pomaze_aliasing.py": sha256_file(
                os.path.join(_HERE, "pomaze_aliasing.py")),
        },
        "decision_rules": prereg["decision_rules"],
    })
    for env_name, cls, mod in (("pomaze", POMaze, "pomaze.py"),
                               ("changing_rule", ChangingRule,
                                "changing_rule.py")):
        em.emit("env_config", {
            "env": env_name, "class": cls.__name__,
            "module_sha256": sha256_file(os.path.join(_ENVS, mod)),
            "wrapper": "experiments/odom_wrapper.py",
            "wrapper_sha256": sha256_file(
                os.path.join(_HERE, "odom_wrapper.py")),
        })

    results = {seed: {} for seed in SEEDS}
    for arm in ARMS:
        for seed in SEEDS:
            agent, eps = run_B_arm(arm, seed, N_EPS)
            results[seed][arm] = eps
            ms = _mean(e["return"] for e in eps)
            pg = sum(1 for e in eps if e["goal"]) / len(eps)
            em.emit("seed_result", {
                "arm": arm, "seed": seed,
                "n_episodes": len(eps),
                "mean_return": round(ms, 4),
                "p_goal": round(pg, 4),
                "n_contexts": sorted({e["n_contexts"] for e in eps}),
                "episodes": eps,
            })
            print(f"  {arm:10s} seed={seed}: mean={ms:+.4f} "
                  f"p_goal={pg:.4f}", flush=True)

    # -- aggregates + frozen decision rules --------------------------------
    def agg(arm):
        ms = {s: _mean(e["return"] for e in results[s][arm]) for s in SEEDS}
        pg = {s: sum(1 for e in results[s][arm] if e["goal"]) / N_EPS
              for s in SEEDS}
        return {"per_seed_mean_return": {str(s): round(ms[s], 4)
                                         for s in SEEDS},
                "per_seed_p_goal": {str(s): round(pg[s], 4) for s in SEEDS},
                "mean_return": round(_mean(ms.values()), 4),
                "p_goal": round(_mean(pg.values()), 4)}

    A = {arm: agg(arm) for arm in ARMS}
    base, aug, shuf = A["B-BASE"], A["B-AUG"], A["B-SHUF"]
    base_cr, aug_cr = A["B-BASE-CR"], A["B-AUG-CR"]

    gate_a = (aug["p_goal"] >= 2 * base["p_goal"] if base["p_goal"] > 0
              else aug["p_goal"] >= 0.20)
    gate_b = aug["mean_return"] - base["mean_return"] >= 0.5
    shuf_a = (shuf["p_goal"] >= 2 * base["p_goal"] if base["p_goal"] > 0
              else shuf["p_goal"] >= 0.20)
    shuf_b = shuf["mean_return"] - base["mean_return"] >= 0.5
    control = aug_cr["mean_return"] - base_cr["mean_return"] >= -1.0

    if gate_a and gate_b:
        verdict = ("H_SUPPORTED" if not (shuf_a and shuf_b)
                   else "H_REJECTED_capacity_confound")
    elif not gate_a and not gate_b:
        verdict = "NULL_HOLDS"
    else:
        verdict = "PARTIAL_INCONCLUSIVE"

    em.emit("verdict", {
        "aggregates": A,
        "delta_mean_return_AUG_minus_BASE": round(
            aug["mean_return"] - base["mean_return"], 4),
        "p_goal_AUG": aug["p_goal"], "p_goal_BASE": base["p_goal"],
        "gate_a_pgoal_doubles": bool(gate_a),
        "gate_b_return_plus05": bool(gate_b),
        "kill_arm_SHUF_gates": {"a": bool(shuf_a), "b": bool(shuf_b)},
        "control_delta_AUGCR_minus_BASECR": round(
            aug_cr["mean_return"] - base_cr["mean_return"], 4),
        "control_gate_pass": bool(control),
        "verdict": verdict,
    })
    print(f"  verdict: {verdict} "
          f"(gate_a={gate_a} gate_b={gate_b} "
          f"shuf=({shuf_a},{shuf_b}) control={control})", flush=True)

    # G0 post: canonical files still identical.
    g0_post = {k: sha256_file(p) for k, p in g0_files.items()}
    g0_ok = all(g0_post[k] == v for k, v in g0_pre.items())

    # G2 determinism: recompute (B-AUG, SEEDS[0]).
    r1 = _mean(e["return"] for e in results[SEEDS[0]]["B-AUG"])
    _, eps2 = run_B_arm("B-AUG", SEEDS[0], N_EPS)
    r2 = _mean(e["return"] for e in eps2)
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "arm": "B-AUG", "seed": SEEDS[0], "first": r1, "rerun": r2,
        "match_1e9": bool(det_ok), "g0_post_match_pre": bool(g0_ok)})
    print(f"  G0 post: {'MATCH' if g0_ok else 'MISMATCH'}; "
          f"G2 determinism: {r1:.6f} vs {r2:.6f} -> "
          f"{'MATCH' if det_ok else 'MISMATCH'}", flush=True)

    ok, msg = em.verify()
    print(f"  chain verify: {ok} ({msg})", flush=True)

    outdir = os.path.join(lab, "receipts")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{EXPERIMENT_ID}.ndjson")
    with open(out, "w") as f:
        for rec in em.records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"  receipt: {out} ({len(em.records)} records, "
          f"{time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
