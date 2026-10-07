"""EXP-FP-0013-RETENTION — causal test of the arch-B goal-trial washout fix.

Preregistered: experiments/preregistration_RETENTION.json (sealed BEFORE
this ran).

Background: EXP-FP-0010 diagnosed B's pomaze failure as credit-RETENTION:
88 goal probes, mean|1.0-rhat| 1.2271->0.9852 transiently, but only
36/88 = 0.41 closer at run end (mean|1.0-rhat_at_end| = 1.1828). Washout
locus (synthetic replay): the per-context R-table entry R_ctx[a*] — the
goal trial adds dR = etaR*piR*rerr ~= +0.94, then every subsequent -0.01
trial re-using action a* self-erases it (rerr ~= -0.01 - R[a*]) within
~50 ticks. Not the episodic store (unbounded, never queried by reward
prediction); not w_r/b_r (track the -0.01 mean by design).

Intervention (ONE, additive): RetentionGuardModel — asymmetric R-table
rate: full etaR on reward > 0 trials, etaR/20 on reward <= 0 trials
(prototypes/architecture-b/retention_guard.py; B's core files untouched).

Arms (4 fresh paired seeds 96001..96004):
  B-BASE      : unmodified ArchB (byte-identical to 0010 make_B)
  B-RETENTION : ArchBRetention, kappa=20.0, goal_thresh=0.0
Envs: canonical pomaze (primary, goal probes), changing_rule +
      delayed_reward (controls, G5: retention seed-mean >= 0.7x base).
15 episodes per (arm, seed, env); one persistent agent per (arm, seed).

Receipts: hash-chained ndjson receipts/EXP-FP-0013-RETENTION.ndjson.
Canonical files untouched (sha256 pre/post, G0). Nothing pushed (G4).

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
from retention_guard import make_retention_agent  # noqa: E402
from env_interface import derive_seed, obs_to_vector  # noqa: E402
from pomaze import POMaze  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402
from delayed_reward import DelayedReward  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0013-RETENTION"
SEEDS = (96001, 96002, 96003, 96004)
N_EPS = 15
KAPPA = 20.0
GOAL_THRESH = 0.0
RETENTION_GATE = 0.70
CONTROL_GATE = 0.7

CANONICAL_FILES = [
    "prototypes/architecture-b/agent.py",
    "prototypes/architecture-b/generative_model.py",
    "prototypes/architecture-b/memory.py",
    "prototypes/architecture-b/precision.py",
    "experiments/envs/pomaze.py",
    "experiments/envs/changing_rule.py",
    "experiments/envs/delayed_reward.py",
]


def make_B(env_obj, agent_seed):
    n_actions = env_obj.action_space()["n"]
    return ArchB(env_obj.observation_space(), n_actions,
                 env_name=env_obj.NAME,
                 action_mode="active_inference", affect="none",
                 seed=agent_seed)


def make_R(env_obj, agent_seed):
    n_actions = env_obj.action_space()["n"]
    return make_retention_agent(
        env_obj.observation_space(), n_actions, env_name=env_obj.NAME,
        seed=agent_seed, kappa=KAPPA, goal_thresh=GOAL_THRESH,
        action_mode="active_inference", affect="none")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_arm(arm, env_cls, primary_seed, n_episodes, collect_probes):
    env = env_cls()
    aseed = derive_seed(primary_seed, 0, "agent")
    agent = make_B(env, aseed) if arm == "B-BASE" else make_R(env, aseed)
    ep_summaries = []
    goal_probes = []
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(ep_seed)
        agent.reset(derive_seed(ep_seed, run_index, "agent"),
                    env.action_space())
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick_n = 0
        done = False
        while not done:
            pos_before = getattr(env, "_pos", None)
            obs_vec = agent._vec(obs)
            a = agent.act(obs)
            rhat_chosen = float(agent._pending["rhat"])
            ctx_now = agent._pending["ctx"]
            obs, r, done, info = env.step(a)
            agent.update(obs, a, r, done, info)
            total += r
            tick_n += 1
            if pos_before is not None and env._pos == pos_before:
                stuck += 1
            if collect_probes and info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick_n
                goal_probes.append({"obs_vec": list(obs_vec), "action": a,
                                    "ctx": list(ctx_now),
                                    "rhat_at_trial": rhat_chosen,
                                    "reward_at_trial": float(r),
                                    "seed": primary_seed,
                                    "ep_idx": run_index})
        ep_summaries.append({"return": round(total, 4), "goal": goal,
                             "steps_to_goal": steps_to_goal,
                             "stuck_frac": round(stuck / max(1, tick_n), 4),
                             "ticks": tick_n})
    if collect_probes:
        model = agent.model
        for gp in goal_probes:
            gp["rhat_at_end"] = float(model.predict_reward(
                gp["obs_vec"], gp["action"], tuple(gp["ctx"])))
    return agent, ep_summaries, goal_probes


# ---------------------------------------------------------------------------
# hash-chained receipt emission (pomaze_diag.py convention)
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


def retention_stats(probes):
    n = len(probes)
    if not n:
        return {"n_probes": 0}
    closer = sum(1 for p in probes
                 if abs(p["reward_at_trial"] - p["rhat_at_end"])
                 < abs(p["reward_at_trial"] - p["rhat_at_trial"]))
    end_err = _mean(abs(p["reward_at_trial"] - p["rhat_at_end"])
                    for p in probes)
    trial_err = _mean(abs(p["reward_at_trial"] - p["rhat_at_trial"])
                      for p in probes)
    return {"n_probes": n, "n_closer": closer,
            "retention_fraction": round(closer / n, 4),
            "mean_abs_err_at_trial": round(trial_err, 4),
            "mean_abs_err_at_end": round(end_err, 4)}


def main():
    t0 = time.time()
    em = Emitter()
    lab = os.path.dirname(_HERE)

    pre_sha = {f: sha256_file(os.path.join(lab, f))
               for f in CANONICAL_FILES}
    with open(os.path.join(_HERE, "preregistration_RETENTION.json")) as f:
        prereg = json.load(f)
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_RETENTION.json",
        "prereg_sha256": hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()).hexdigest(),
        "experiment_id": EXPERIMENT_ID,
        "type": prereg["type"],
        "seeds": list(SEEDS),
        "arms": ["B-BASE", "B-RETENTION"],
        "kappa": KAPPA, "goal_thresh": GOAL_THRESH,
        "canonical_sha256_pre": pre_sha,
        "hypothesis": prereg["hypothesis"],
        "null_hypothesis": prereg["null_hypothesis"],
    })

    envs = [("pomaze", POMaze, True),
            ("changing_rule", ChangingRule, False),
            ("delayed_reward", DelayedReward, False)]
    results = {}  # results[env][arm][seed] = {"episodes", "goal_probes"}
    for env_name, env_cls, probes in envs:
        results[env_name] = {}
        for arm in ("B-BASE", "B-RETENTION"):
            results[env_name][arm] = {}
            for seed in SEEDS:
                _, eps, gp = run_arm(arm, env_cls, seed, N_EPS, probes)
                results[env_name][arm][seed] = {"episodes": eps,
                                                "goal_probes": gp}
                ms = _mean(e["return"] for e in eps)
                pg = sum(1 for e in eps if e["goal"])
                em.emit("seed_result", {
                    "env": env_name, "arm": arm, "seed": seed,
                    "n_episodes": len(eps),
                    "mean_return": round(ms, 4),
                    "p_goal": round(pg / max(1, len(eps)), 4),
                    "episodes": eps,
                    "n_goal_probes": len(gp),
                    "goal_probes": [
                        {k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in p.items() if k != "obs_vec"}
                        for p in gp],
                })
                print(f"  {env_name:14s} {arm:11s} seed={seed}: "
                      f"mean={ms:+.4f} p_goal={pg}/{len(eps)} "
                      f"probes={len(gp)}", flush=True)

    # -- verdict ---------------------------------------------------------
    base_probes = [p for s in SEEDS
                   for p in results["pomaze"]["B-BASE"][s]["goal_probes"]]
    ret_probes = [p for s in SEEDS
                  for p in results["pomaze"]["B-RETENTION"][s]["goal_probes"]]
    base_rs = retention_stats(base_probes)
    ret_rs = retention_stats(ret_probes)

    def arm_pgoal(env, arm):
        return {s: sum(1 for e in results[env][arm][s]["episodes"]
                       if e["goal"]) / N_EPS for s in SEEDS}

    def arm_meanret(env, arm):
        return {s: _mean(e["return"]
                         for e in results[env][arm][s]["episodes"])
                for s in SEEDS}

    verdict = {"retention_B-BASE": base_rs,
               "retention_B-RETENTION": ret_rs,
               "baseline_retention_fraction_0010": 0.41,
               "gate_retention": RETENTION_GATE}
    verdict["primary_gate_pass"] = bool(
        ret_rs.get("retention_fraction", 0.0) >= RETENTION_GATE)
    verdict["p_goal_pomaze"] = {
        "B-BASE": {str(s): v for s, v in arm_pgoal("pomaze", "B-BASE").items()},
        "B-RETENTION": {str(s): v
                        for s, v in arm_pgoal("pomaze",
                                              "B-RETENTION").items()}}
    verdict["mean_return_pomaze"] = {
        "B-BASE": {str(s): round(v, 4)
                   for s, v in arm_meanret("pomaze", "B-BASE").items()},
        "B-RETENTION": {str(s): round(v, 4)
                        for s, v in arm_meanret("pomaze",
                                                "B-RETENTION").items()}}
    controls = {}
    for env_name in ("changing_rule", "delayed_reward"):
        b = arm_meanret(env_name, "B-BASE")
        r = arm_meanret(env_name, "B-RETENTION")
        mb, mr = _mean(b.values()), _mean(r.values())
        gate = (mr >= CONTROL_GATE * mb) if mb > 0 else (mr >= mb)
        controls[env_name] = {
            "B-BASE_seed_mean": round(mb, 4),
            "B-RETENTION_seed_mean": round(mr, 4),
            "gate_0.7x": bool(gate),
            "per_seed": {str(s): {"base": round(b[s], 4),
                                  "ret": round(r[s], 4)} for s in SEEDS}}
    verdict["controls"] = controls
    verdict["controls_pass"] = all(c["gate_0.7x"]
                                   for c in controls.values())
    if not verdict["controls_pass"]:
        verdict["decision"] = "REJECTED_CONTROL_DAMAGE"
    elif verdict["primary_gate_pass"]:
        verdict["decision"] = "H_SUPPORTED"
    else:
        verdict["decision"] = "NULL_HOLDS"
    em.emit("verdict", verdict)
    print("  verdict: " + json.dumps(
        {k: v for k, v in verdict.items()
         if k in ("decision", "primary_gate_pass", "controls_pass")}),
        flush=True)

    # -- G2 determinism: recompute (B-RETENTION, pomaze, 96001) -----------
    r1 = _mean(e["return"]
               for e in results["pomaze"]["B-RETENTION"][96001]["episodes"])
    _, eps2, _ = run_arm("B-RETENTION", POMaze, 96001, N_EPS, True)
    r2 = _mean(e["return"] for e in eps2)
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "arm": "B-RETENTION", "env": "pomaze", "seed": 96001,
        "first": r1, "rerun": r2, "match_1e9": bool(det_ok)})
    print(f"  G2 determinism: {r1:.6f} vs {r2:.6f} -> "
          f"{'MATCH' if det_ok else 'MISMATCH'}", flush=True)

    # -- G0 post-run sha check -------------------------------------------
    post_sha = {f: sha256_file(os.path.join(lab, f))
                for f in CANONICAL_FILES}
    g0_ok = all(post_sha[f] == pre_sha[f] for f in CANONICAL_FILES)
    em.emit("additivity_check", {
        "canonical_files": CANONICAL_FILES,
        "pre_match_post": bool(g0_ok),
        "post_sha256": post_sha})
    print(f"  G0 additivity: {'MATCH' if g0_ok else 'MISMATCH'}", flush=True)

    ok, msg = em.verify()
    print(f"  chain verify: {ok} ({msg})", flush=True)

    out = os.path.join(lab, "receipts", f"{EXPERIMENT_ID}.ndjson")
    with open(out, "w") as f:
        for rec in em.records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"  receipt: {out} ({len(em.records)} records, "
          f"{time.time() - t0:.1f}s)", flush=True)
    if not (det_ok and g0_ok and ok):
        raise SystemExit("FROZEN GATE FAILURE")


if __name__ == "__main__":
    main()
