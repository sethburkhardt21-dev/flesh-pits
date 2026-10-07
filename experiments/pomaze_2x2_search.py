"""EXP-FP-0012-2X2-SEARCH — disambiguation x systematic-search factorial on pomaze.

Preregistered: experiments/preregistration_2X2_SEARCH.json (sealed BEFORE run).

Resolving follow-up to EXP-FP-0011-ALIASING (null held: odometry
disambiguation alone did NOT unlock B) per the recorded mechanism lead:
"representation can only bind once reward is found repeatedly — next
resolving experiment pairs disambiguation with a systematic-search
exploration intervention."

2x2 factorial on canonical POMaze, arch-B only (config byte-identical to
EXP-FP-0010 make_B / EXP-FP-0011 run_B_arm):
  B-BASE      : unmodified B (baseline replication)
  B-AUG       : B + odometry disambiguation (odom_wrapper.py, drive='true',
                reused VERBATIM — sha matches the EXP-FP-0011 receipt)
  B-SEARCH    : B + systematic-search exploration intervention ONLY
  B-AUGSEARCH : BOTH

SEARCH intervention (additive, driver-side, frozen in the preregistration):
count-based coverage bonus on (obs_key, action) visit counts. The key uses
ONLY the 5 canonical pomaze obs fields (wall_n/s/e/w, beacon) — it NEVER
reads odom_qx/odom_qy (present but ignored in AUG arms) and never reads env
internals, goal position, or walls. Same key function in B-SEARCH and
B-AUGSEARCH, so the AUG x SEARCH interaction cannot be confounded by the
search mechanism reading the disambiguation channel. Counts accumulate
across all 15 episodes of a run. Bonus b(a) = beta / sqrt(1 + c[(key,a)]),
beta = 0.3 (reward units), added to the selector's vhat inside a
per-instance wrap of agent.selector.select. B's core files are untouched.

Preregistered hypothesis: the INTERACTION —
  (p_goal_AS - p_goal_S) - (p_goal_A - p_goal_B) >= 0.15  AND
  (ret_AS - ret_S) - (ret_A - ret_B) >= 0.5
i.e. disambiguation helps ONLY once search finds reward repeatedly.
Null: no interaction. Main effects + manipulation checks + binding
diagnostics are descriptive (no gates).

Receipts: hash-chained ndjson receipts/EXP-FP-0012-2X2-SEARCH.ndjson.
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
from odom_wrapper import OdomWrapper  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0012-2X2-SEARCH"
SEEDS = (94007, 94008, 94009, 94010, 94011, 94012)
N_EPS = 15
BETA = 0.3  # frozen in preregistration_2X2_SEARCH.json

ARMS = ["B-BASE", "B-AUG", "B-SEARCH", "B-AUGSEARCH"]
KEY_FIELDS = ("wall_n", "wall_s", "wall_e", "wall_w", "beacon")


def obs_key(obs):
    """Frozen SEARCH key: 5 canonical pomaze fields only. Never reads
    odom_qx/odom_qy (even when present), never reads env internals."""
    return tuple(round(float(obs[f]), 3) for f in KEY_FIELDS)


class CountBonusDriver:
    """Additive systematic-search exploration for ArchB.

    Wraps agent.selector.select at the instance level: adds
    beta / sqrt(1 + c[(key, action)]) to the selector's vhat for the
    current tick. Everything else about the agent (weights, beliefs,
    memory, learning, selector lambdas/error tables) is untouched.
    """

    def __init__(self, agent, beta=BETA):
        self.agent = agent
        self.beta = beta
        self.counts = {}
        self._key = None
        self.last_bonus = 0.0
        orig_select = agent.selector.select

        def select(predict_reward, predicted_error, risk):
            key = self._key
            beta = self.beta
            counts = self.counts

            def vhat_bonus(a):
                v = predict_reward(a)
                if key is not None:
                    v += beta / math.sqrt(
                        1.0 + counts.get((key, a), 0))
                return v

            chosen = orig_select(vhat_bonus, predicted_error, risk)
            if key is not None:
                self.last_bonus = beta / math.sqrt(
                    1.0 + counts.get((key, chosen), 0))
            else:
                self.last_bonus = 0.0
            return chosen

        agent.selector.select = select

    def note_key(self, key):
        self._key = key

    def register(self, key, action):
        k = (key, action)
        self.counts[k] = self.counts.get(k, 0) + 1

    @property
    def coverage(self):
        return len(self.counts)


def make_env(arm):
    """Returns (step_env, raw_env, search: bool)."""
    if arm == "B-BASE":
        env = POMaze()
        return env, env, False
    if arm == "B-AUG":
        wrap = OdomWrapper(POMaze(), drive="true")
        return wrap, wrap.env, False
    if arm == "B-SEARCH":
        env = POMaze()
        return env, env, True
    if arm == "B-AUGSEARCH":
        wrap = OdomWrapper(POMaze(), drive="true")
        return wrap, wrap.env, True
    raise ValueError(f"unknown arm {arm!r}")


def run_B_arm(arm, primary_seed, n_episodes):
    """ArchB closed-loop. Returns (agent, ep_summaries, goal_probes, driver).
    Agent config byte-identical to EXP-FP-0010 make_B / EXP-FP-0011."""
    env, raw, search = make_env(arm)
    space = env.observation_space()
    n_actions = env.action_space()["n"]
    agent = ArchB(space, n_actions, env_name=env.NAME,
                  action_mode="active_inference", affect="none",
                  seed=derive_seed(primary_seed, 0, "agent"))
    driver = CountBonusDriver(agent) if search else None
    ep_summaries = []
    goal_probes = []
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        if isinstance(env, OdomWrapper):
            obs = env.reset(ep_seed, run_index)
        else:
            obs = env.reset(ep_seed)
        agent.reset(derive_seed(ep_seed, run_index, "agent"),
                    env.action_space())
        rerrs, rhats, bonuses = [], [], []
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick_n = 0
        done = False
        while not done:
            pos_before = getattr(raw, "_pos", None)
            key = obs_key(obs)
            if driver is not None:
                driver.note_key(key)
            a = agent.act(obs)
            state_vec = list(agent.model.mu0)
            ctx = agent._pending["ctx"]
            rhat_chosen = float(agent._pending["rhat"])
            obs, r, done, info = env.step(a)
            agent.update(obs, a, r, done, info)
            if driver is not None:
                driver.register(key, a)
                bonuses.append(driver.last_bonus)
            rerrs.append(abs(r - rhat_chosen))
            rhats.append(rhat_chosen)
            total += r
            tick_n += 1
            if pos_before is not None and getattr(raw, "_pos", None) \
                    == pos_before:
                stuck += 1
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick_n
                goal_probes.append({
                    "state_vec": state_vec, "action": a,
                    "ctx": tuple(ctx),
                    "rhat_at_trial": rhat_chosen,
                    "reward_at_trial": float(r),
                    "seed": primary_seed, "ep_idx": run_index,
                })
            if done:
                break
        ep_summaries.append({
            "return": round(total, 4), "goal": goal,
            "steps_to_goal": steps_to_goal,
            "stuck_frac": (round(stuck / max(1, tick_n), 4)
                           if pos_before is not None else None),
            "ticks": tick_n,
            "mean_abs_rerr": round(sum(rerrs) / max(1, len(rerrs)), 5),
            "mean_rhat_chosen": round(sum(rhats) / max(1, len(rhats)), 5),
            "mean_abs_rhat_chosen": round(
                sum(abs(x) for x in rhats) / max(1, len(rhats)), 5),
            "n_contexts": agent.model.n_contexts,
            "coverage_pairs": driver.coverage if driver else None,
            "mean_bonus": (round(sum(bonuses) / max(1, len(bonuses)), 5)
                           if driver else None),
        })
    # Final-model re-evaluation of every goal probe (D7-style retention).
    model = agent.model
    for gp in goal_probes:
        gp["rhat_at_end"] = float(model.predict_reward(
            gp["state_vec"], gp["action"], tuple(gp["ctx"])))
    return agent, ep_summaries, goal_probes, driver


# ---------------------------------------------------------------------------
# hash-chained receipt emission (K10 / pomaze_aliasing convention)
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
        "experiments/env_interface.py": os.path.join(_HERE,
                                                     "env_interface.py"),
        "prototypes/architecture-b/agent.py": os.path.join(
            _ARCH_B, "agent.py"),
        "prototypes/architecture-b/generative_model.py": os.path.join(
            _ARCH_B, "generative_model.py"),
        "prototypes/architecture-b/active_inference.py": os.path.join(
            _ARCH_B, "active_inference.py"),
    }
    g0_pre = {k: sha256_file(p) for k, p in g0_files.items()}
    with open(os.path.join(_HERE, "preregistration_2X2_SEARCH.json")) as f:
        prereg = json.load(f)
    odom_sha = sha256_file(os.path.join(_HERE, "odom_wrapper.py"))
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_2X2_SEARCH.json",
        "prereg_sha256": hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()).hexdigest(),
        "experiment_id": EXPERIMENT_ID,
        "type": prereg["type"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "episodes_per_arm_per_seed": N_EPS,
        "beta": BETA,
        "g0_sha256_pre": g0_pre,
        "new_files_sha256": {
            "experiments/pomaze_2x2_search.py": sha256_file(
                os.path.join(_HERE, "pomaze_2x2_search.py")),
        },
        "odom_wrapper_sha256": odom_sha,
        "odom_wrapper_verbatim_vs_EXP_FP_0011":
            "fe5c6464a94310ad1e09cc5de89f62c8ae990f76fddb3fff7989dd065d76b1c9"
            == odom_sha,
        "decision_rules": prereg["decision_rules"],
    })
    em.emit("env_config", {
        "env": "pomaze", "class": "POMaze",
        "module_sha256": sha256_file(os.path.join(_ENVS, "pomaze.py")),
        "wrappers": ["experiments/odom_wrapper.py (drive='true', verbatim)",
                     "CountBonusDriver (in-driver, beta=0.3, 5-field key)"],
        "wrapper_sha256": odom_sha,
    })

    results = {seed: {} for seed in SEEDS}
    probes = {seed: {} for seed in SEEDS}
    for arm in ARMS:
        for seed in SEEDS:
            agent, eps, gp, driver = run_B_arm(arm, seed, N_EPS)
            results[seed][arm] = eps
            probes[seed][arm] = gp
            ms = _mean(e["return"] for e in eps)
            pg = sum(1 for e in eps if e["goal"]) / len(eps)
            em.emit("seed_result", {
                "arm": arm, "seed": seed,
                "n_episodes": len(eps),
                "mean_return": round(ms, 4),
                "p_goal": round(pg, 4),
                "n_contexts": sorted({e["n_contexts"] for e in eps}),
                "coverage_pairs_final": (eps[-1]["coverage_pairs"]
                                         if driver else None),
                "n_goal_probes": len(gp),
                "episodes": eps,
                "goal_probes": [
                    {k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in p.items()
                     if k not in ("state_vec",)}
                    for p in gp],
            })
            print(f"  {arm:12s} seed={seed}: mean={ms:+.4f} "
                  f"p_goal={pg:.4f} probes={len(gp)}", flush=True)

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
    pB, pA = A["B-BASE"]["p_goal"], A["B-AUG"]["p_goal"]
    pS, pAS = A["B-SEARCH"]["p_goal"], A["B-AUGSEARCH"]["p_goal"]
    rB, rA = A["B-BASE"]["mean_return"], A["B-AUG"]["mean_return"]
    rS, rAS = A["B-SEARCH"]["mean_return"], A["B-AUGSEARCH"]["mean_return"]
    int_p = (pAS - pS) - (pA - pB)
    int_r = (rAS - rS) - (rA - rB)
    gate_a = int_p >= 0.15
    gate_b = int_r >= 0.5
    if gate_a and gate_b:
        verdict = "H_SUPPORTED"
    elif not gate_a and not gate_b:
        verdict = "NULL_HOLDS"
    else:
        verdict = "PARTIAL_INCONCLUSIVE"

    main_search_p = (pS + pAS) / 2 - (pB + pA) / 2
    main_aug_p = (pA + pAS) / 2 - (pB + pS) / 2
    main_search_r = (rS + rAS) / 2 - (rB + rA) / 2
    main_aug_r = (rA + rAS) / 2 - (rB + rS) / 2

    # manipulation checks (descriptive)
    def arm_cov(arm):
        return _mean(results[s][arm][-1]["coverage_pairs"] or 0
                     for s in SEEDS)

    def arm_bonus(arm):
        bs = [e["mean_bonus"] for s in SEEDS
              for e in results[s][arm] if e["mean_bonus"] is not None]
        return round(_mean(bs), 5) if bs else None

    def arm_absrhat(arm):
        return round(_mean(e["mean_abs_rhat_chosen"]
                           for s in SEEDS for e in results[s][arm]), 5)

    # binding diagnostics (descriptive): goal-probe retention per arm
    def probe_retention(arm):
        allp = [p for s in SEEDS for p in probes[s][arm]]
        if not allp:
            return {"n_probes": 0}
        d_trial = [abs(p["rhat_at_trial"] - p["reward_at_trial"])
                   for p in allp]
        d_end = [abs(p["rhat_at_end"] - p["reward_at_trial"])
                 for p in allp]
        return {"n_probes": len(allp),
                "mean_abs_rerr_at_trial": round(_mean(d_trial), 4),
                "mean_abs_rerr_at_end": round(_mean(d_end), 4),
                "frac_closer_at_end": round(
                    sum(1 for a, b in zip(d_trial, d_end) if b < a)
                    / len(allp), 4)}

    def late_early(arm):
        gaps = []
        for s in SEEDS:
            eps = results[s][arm]
            early = sum(1 for e in eps[:7] if e["goal"]) / 7
            late = sum(1 for e in eps[7:] if e["goal"]) / 8
            gaps.append(late - early)
        return round(_mean(gaps), 4)

    verdict_rec = {
        "aggregates": A,
        "interaction_p_goal": round(int_p, 4),
        "interaction_mean_return": round(int_r, 4),
        "gate_a_interaction_pgoal_ge_015": bool(gate_a),
        "gate_b_interaction_return_ge_05": bool(gate_b),
        "verdict": verdict,
        "main_effects_descriptive": {
            "search_main_p_goal": round(main_search_p, 4),
            "aug_main_p_goal": round(main_aug_p, 4),
            "search_main_mean_return": round(main_search_r, 4),
            "aug_main_mean_return": round(main_aug_r, 4),
        },
        "manipulation_checks_descriptive": {
            "mean_coverage_pairs": {arm: round(arm_cov(arm), 1)
                                    for arm in ARMS},
            "mean_bonus_per_tick": {arm: arm_bonus(arm) for arm in ARMS},
            "mean_abs_rhat_chosen": {arm: arm_absrhat(arm) for arm in ARMS},
        },
        "binding_diagnostics_descriptive": {
            "goal_probe_retention": {arm: probe_retention(arm)
                                     for arm in ARMS},
            "late_minus_early_p_goal": {arm: late_early(arm)
                                        for arm in ARMS},
        },
    }
    em.emit("verdict", verdict_rec)
    print(f"  verdict: {verdict} (int_p={int_p:+.4f}>=0.15? {gate_a} "
          f"int_r={int_r:+.4f}>=0.5? {gate_b})", flush=True)
    print(f"  main effects: search p {main_search_p:+.4f} / ret "
          f"{main_search_r:+.4f}; aug p {main_aug_p:+.4f} / ret "
          f"{main_aug_r:+.4f}", flush=True)

    # G0 post: canonical files still identical.
    g0_post = {k: sha256_file(p) for k, p in g0_files.items()}
    g0_ok = all(g0_post[k] == v for k, v in g0_pre.items())

    # G2 determinism: recompute (B-AUGSEARCH, SEEDS[0]).
    r1 = _mean(e["return"] for e in results[SEEDS[0]]["B-AUGSEARCH"])
    _, eps2, _, _ = run_B_arm("B-AUGSEARCH", SEEDS[0], N_EPS)
    r2 = _mean(e["return"] for e in eps2)
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "arm": "B-AUGSEARCH", "seed": SEEDS[0], "first": r1, "rerun": r2,
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
