"""EXP-FP-0010-POMAZE-DIAG — cross-architecture pomaze failure diagnosis.

DIAGNOSTIC (characterization, not a kill). Preregistered:
  experiments/preregistration_POMAZE_DIAG.json  (sealed BEFORE this ran)

Question: why does pomaze defeat every arm? For each architecture
(A: workspace/bandit, B: predictive model, C: hybrid), is the failure
EXPLORATION (never finds reward), CREDIT ASSIGNMENT (finds but doesn't
learn), or REPRESENTATION (learns but can't use)?

Arms (6 fresh primary seeds 91001..91006, paired):
  RANDOM / SYMBOLIC : reward availability + expert ceiling (canonical)
  {A,B,C}-STD       : intact battery configs on canonical pomaze
  {A,B,C}-DENSE     : pomaze_dense (additive +0.02*beacon shaping)
  {A,B,C}-NEAR      : pomaze_near (additive near-goal start, dist<=4)
  {A,B,C}-DEMO      : 5 NEAR episodes then 15 canonical (persistent agent)

Telemetry per episode: return, goal, steps_to_goal, stuck fraction;
  A/C: gains trajectory, tie-break fraction (margin<1e-9), stimulus/obs
       correlation, stimulus stream std;
  B: rhat/rerr per tick, goal-step rhat probes, |w_r|, R-table, n_contexts.
D8 aliasing: from RANDOM rollouts, obs-key -> positions -> BFS-optimal
  actions (driver-side analysis, not policy).

Receipts: hash-chained ndjson receipts/EXP-FP-0010-POMAZE-DIAG.ndjson.
Canonical pomaze.py untouched (sha256 recorded, G0). Nothing pushed (G4).

Stdlib only. Deterministic given seeds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_C = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-c")
_ARCH_A = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-a")
_ARCH_B = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-b")
_ENVS = os.path.join(_HERE, "envs")
_BASELINES = os.path.join(_HERE, "baselines")
_AEXP = os.path.join(_ARCH_A, "experiments")
for _p in (_ARCH_C, _ARCH_A, _ARCH_B, _HERE, _ENVS, _BASELINES, _AEXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_agent import ArchC, CAdapter  # noqa: E402
from tick import WorkspaceTick  # noqa: E402
from attention_cue import CueIndexedArbitrator  # noqa: E402
from k5_multitask_generalization import neutral_specialists  # noqa: E402
from agent import ArchB  # noqa: E402
from env_interface import derive_seed, new_rng, obs_to_vector  # noqa: E402
from pomaze import POMaze  # noqa: E402
from pomaze_dense import POMazeDense  # noqa: E402
from pomaze_near import POMazeNear  # noqa: E402
from symbolic_baseline import SymbolicBaselineAgent  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0010-POMAZE-DIAG"
SEEDS = (91001, 91002, 91003, 91004, 91005, 91006)
THETA = 0.45
GAIN_LR = 0.15
CHANNELS = ["north", "south", "east", "west"]
C2A = {"north": 0, "south": 1, "east": 2, "west": 3}
N_EPS_BASE = 30   # RANDOM / SYMBOLIC
N_EPS_ARCH = 15   # STD / DENSE / NEAR
N_EPS_DEMO_NEAR = 5
DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W


# ---------------------------------------------------------------------------
# agent constructors — byte-identical config to EXP-FP-C-BUILD-AND-BEAT
# ---------------------------------------------------------------------------
def make_A(agent_seed):
    tick = WorkspaceTick(
        CHANNELS, neutral_specialists(CHANNELS, agent_seed),
        capacity=len(CHANNELS), frozen_gains=False, gain_lr=GAIN_LR,
        ignition_kwargs={"theta": THETA})
    return tick


def make_B(env_obj, agent_seed):
    return ArchB(env_obj.observation_space(), len(CHANNELS),
                 env_name=env_obj.NAME,
                 action_mode="active_inference", affect="none",
                 seed=agent_seed)


def make_C(env_obj, agent_seed):
    cfg = {
        "channels": CHANNELS, "channel_to_action": C2A,
        "observation_space": env_obj.observation_space(),
        "env_name": env_obj.NAME, "context_fn": None,
        "gain_lr": GAIN_LR, "theta": THETA, "kappa": 0.1,
        "agent_seed": agent_seed,
        "predictor_kwargs": {"eta0": 0.005, "etaD": 0.002, "eta_r": 0.05,
                             "etaR": 0.10, "precision_kind": "estimated",
                             "max_contexts": 32, "precision_window": 50},
        "memory_enabled": True, "frozen_predictor": False,
        "frozen_gains": False,
    }
    return ArchC(cfg)


def gains_of(agent, arch):
    if arch == "B":
        return None
    tick = agent if arch == "A" else agent.tick
    return dict(tick.arbitrator.gains)


# ---------------------------------------------------------------------------
# BFS optimal action (driver-side analysis for the D8 aliasing measure)
# ---------------------------------------------------------------------------
def bfs_first_action(walls, start, goal):
    if start == goal:
        return None
    prev = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for ai, (dx, dy) in enumerate(DIRS):
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in prev or nb in walls:
                continue
            prev[nb] = (cur, ai)
            if nb == goal:
                # walk back to the first step
                node = nb
                while prev[node][0] != start:
                    node = prev[node][0]
                return prev[node][1]
            dq.append(nb)
    return None


# ---------------------------------------------------------------------------
# episode runners
# ---------------------------------------------------------------------------
def _obs_key(obs):
    return (obs["wall_n"], obs["wall_s"], obs["wall_e"], obs["wall_w"],
            obs["beacon"])


def run_random(env_cls, primary_seed, alias_collect):
    """Uniform random policy. alias_collect: list to append
    (obs_key, pos, optimal_action) per tick for the D8 measure."""
    env = env_cls()
    ep_summaries = []
    for run_index in range(N_EPS_BASE):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(ep_seed)
        rng = new_rng(derive_seed(primary_seed, run_index, "agent"))
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick = 0
        while True:
            pos_before = env._pos
            key = _obs_key(obs)
            opt = bfs_first_action(env._walls, pos_before, env._goal)
            alias_collect.append((key, pos_before, opt))
            a = rng.randrange(4)
            obs, r, done, info = env.step(a)
            total += r
            tick += 1
            if env._pos == pos_before:
                stuck += 1
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick
            if done:
                break
        ep_summaries.append({"return": round(total, 4), "goal": goal,
                             "steps_to_goal": steps_to_goal,
                             "stuck_frac": round(stuck / max(1, tick), 4),
                             "ticks": tick})
    return ep_summaries


def run_symbolic(env_cls, primary_seed):
    env = env_cls()
    agent = SymbolicBaselineAgent()
    ep_summaries = []
    for run_index in range(N_EPS_BASE):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(ep_seed)
        agent.reset(derive_seed(primary_seed, run_index, "agent"),
                    env.action_space())
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick = 0
        while True:
            pos_before = env._pos
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            agent.update(obs, a, r, done, info)
            total += r
            tick += 1
            if env._pos == pos_before:
                stuck += 1
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick
            if done:
                break
        ep_summaries.append({"return": round(total, 4), "goal": goal,
                             "steps_to_goal": steps_to_goal,
                             "stuck_frac": round(stuck / max(1, tick), 4),
                             "ticks": tick})
    return ep_summaries


def run_tick_arm(arch, env_cls, primary_seed, n_episodes, agent=None,
                 collect_probes=None):
    """A/C via the CAdapter protocol (run_battery convention). Returns
    (agent, ep_summaries). collect_probes: optional list for B-style goal
    probes — unused here (B has its own runner)."""
    env = env_cls()
    if agent is None:
        aseed = derive_seed(primary_seed, 0, "agent")
        agent = make_A(aseed) if arch == "A" else make_C(env, aseed)
    ep_summaries = []
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        adapter = CAdapter(env, C2A)
        obs = adapter.reset_episode(ep_seed)
        if arch == "C":
            agent.reset_episode(obs)
        g0 = gains_of(agent, arch)
        margins, stimuli_stream, beacon_stream = [], [], []
        stim_by_channel = {c: [] for c in CHANNELS}
        sub_ign = 0
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick_n = 0
        while True:
            pos_before = env._pos
            if arch == "A":
                trace = agent.step(obs, adapter)
            else:
                trace = agent.step(adapter)
            r = trace["reward"]
            total += r
            tick_n += 1
            arb = trace["arbitration"]
            margins.append(arb["margin"])
            if trace.get("action_selection", {}).get("path") == \
                    "sub_ignition_explore":
                sub_ign += 1
            stims = trace["stimuli"]
            stimuli_stream.append(sum(stims.values()) / len(stims))
            for c in CHANNELS:
                stim_by_channel[c].append(stims[c])
            beacon_stream.append(obs["beacon"])
            if env._pos == pos_before:
                stuck += 1
            _obs_next, _r, done, info = adapter.last_transition()
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick_n
            if done:
                break
            obs = adapter.observe()
        g1 = gains_of(agent, arch)
        tie_frac = sum(1 for m in margins if m < 1e-9) / max(1, len(margins))
        mean_s = sum(stimuli_stream) / max(1, len(stimuli_stream))
        var_s = (sum((s - mean_s) ** 2 for s in stimuli_stream)
                 / max(1, len(stimuli_stream)))
        corrs = {}
        for c in CHANNELS:
            corrs[c] = round(_pearson(stim_by_channel[c], beacon_stream), 4)
        ep = {"return": round(total, 4), "goal": goal,
              "steps_to_goal": steps_to_goal,
              "stuck_frac": round(stuck / max(1, tick_n), 4),
              "ticks": tick_n,
              "tie_break_frac": round(tie_frac, 4),
              "sub_ignition_frac": round(sub_ign / max(1, tick_n), 4),
              "stimuli_std": round(math.sqrt(var_s), 5),
              "stim_beacon_corr": corrs,
              "gains_start": {k: round(v, 4) for k, v in g0.items()},
              "gains_end": {k: round(v, 4) for k, v in g1.items()},
              "margin_mean": round(sum(margins) / max(1, len(margins)), 6)}
        if arch == "C":
            ep["n_contexts"] = agent.predictor.n_contexts
            ep["update_norm_mean"] = None  # filled below if wanted
        ep_summaries.append(ep)
    return agent, ep_summaries


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def run_B_arm(env_cls, primary_seed, n_episodes, agent=None):
    """ArchB with per-tick rhat capture. Returns (agent, ep_summaries,
    goal_probes) where goal_probes = [(obs_vec, action, rhat_at_trial,
    reward_at_trial, seed, ep_idx)]."""
    env = env_cls()
    if agent is None:
        agent = make_B(env, derive_seed(primary_seed, 0, "agent"))
    ep_summaries = []
    goal_probes = []
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(ep_seed)
        agent.reset(derive_seed(ep_seed, run_index, "agent"),
                    env.action_space())
        rerrs, rhats = [], []
        total, stuck, goal, steps_to_goal = 0.0, 0, False, None
        tick_n = 0
        done = False
        while not done:
            pos_before = env._pos
            obs_vec = agent._vec(obs)
            a = agent.act(obs)
            rhat_chosen = float(agent._pending["rhat"])
            ctx_now = agent._pending["ctx"]
            obs, r, done, info = env.step(a)
            agent.update(obs, a, r, done, info)
            rerrs.append(abs(r - rhat_chosen))
            rhats.append(rhat_chosen)
            total += r
            tick_n += 1
            if env._pos == pos_before:
                stuck += 1
            if info.get("goal_reached") and not goal:
                goal, steps_to_goal = True, tick_n
                goal_probes.append({"obs_vec": list(obs_vec), "action": a,
                                    "ctx": list(ctx_now),
                                    "rhat_at_trial": rhat_chosen,
                                    "reward_at_trial": float(r),
                                    "seed": primary_seed,
                                    "ep_idx": run_index})
        model = agent.model
        wr = model.w_r
        wr_norm = math.sqrt(sum(w * w for w in wr))
        r_table = model.ctx_R.get((), [None] * 4)
        ep_summaries.append({
            "return": round(total, 4), "goal": goal,
            "steps_to_goal": steps_to_goal,
            "stuck_frac": round(stuck / max(1, tick_n), 4),
            "ticks": tick_n,
            "mean_abs_rerr": round(sum(rerrs) / max(1, len(rerrs)), 5),
            "max_abs_rerr": round(max(rerrs) if rerrs else 0.0, 4),
            "mean_rhat_chosen": round(sum(rhats) / max(1, len(rhats)), 5),
            "wr_norm": round(wr_norm, 5),
            "R_table": [round(v, 4) if v is not None else None
                        for v in r_table],
            "n_contexts": model.n_contexts,
        })
    # Re-evaluate every goal-step probe with the FINAL model: how much did
    # the reward head move toward the observed reward? (D7)
    for gp in goal_probes:
        gp["rhat_at_end"] = float(model.predict_reward(
            gp["obs_vec"], gp["action"], tuple(gp["ctx"])))
    return agent, ep_summaries, goal_probes


# ---------------------------------------------------------------------------
# arm orchestration
# ---------------------------------------------------------------------------
def run_arm(arm, primary_seed, alias_collect):
    """Returns dict with per-episode summaries + arm-level telemetry."""
    if arm == "RANDOM":
        eps = run_random(POMaze, primary_seed, alias_collect)
        return {"episodes": eps, "goal_probes": []}
    if arm == "SYMBOLIC":
        eps = run_symbolic(POMaze, primary_seed)
        return {"episodes": eps, "goal_probes": []}
    arch, variant = arm.split("-", 1)
    if variant == "DEMO":
        env_cls = None  # DEMO runs NEAR then STD explicitly below
    else:
        env_cls = {"STD": POMaze, "DENSE": POMazeDense,
                   "NEAR": POMazeNear}[variant]
    if arch == "B":
        if variant == "DEMO":
            agent, near_eps, near_probes = run_B_arm(
                POMazeNear, primary_seed, N_EPS_DEMO_NEAR)
            agent, eps, probes = run_B_arm(
                POMaze, primary_seed, N_EPS_ARCH, agent=agent)
            return {"episodes": eps, "near_episodes": near_eps,
                    "goal_probes": near_probes + probes}
        agent, eps, probes = run_B_arm(env_cls, primary_seed, N_EPS_ARCH)
        return {"episodes": eps, "goal_probes": probes}
    # A / C tick arms
    if variant == "DEMO":
        agent, near_eps = run_tick_arm(arch, POMazeNear, primary_seed,
                                       N_EPS_DEMO_NEAR)
        agent, eps = run_tick_arm(arch, POMaze, primary_seed, N_EPS_ARCH,
                                  agent=agent)
        return {"episodes": eps, "near_episodes": near_eps,
                "goal_probes": []}
    agent, eps = run_tick_arm(arch, env_cls, primary_seed, N_EPS_ARCH)
    return {"episodes": eps, "goal_probes": []}


ARMS = ["RANDOM", "SYMBOLIC",
        "A-STD", "B-STD", "C-STD",
        "A-DENSE", "B-DENSE", "C-DENSE",
        "A-NEAR", "B-NEAR", "C-NEAR",
        "A-DEMO", "B-DEMO", "C-DEMO"]


# ---------------------------------------------------------------------------
# hash-chained receipt emission (K10 / run_battery convention)
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


def arm_means(results, arm):
    """results[seed][arm] -> list of per-episode returns."""
    out = {}
    for seed in SEEDS:
        out[seed] = _mean(e["return"]
                          for e in results[seed][arm]["episodes"])
    return out


def p_goal(results, arm):
    eps = [e for seed in SEEDS for e in results[seed][arm]["episodes"]]
    return sum(1 for e in eps if e["goal"]) / max(1, len(eps))


# ---------------------------------------------------------------------------
# D1..D11 discriminators + failure-class verdict rules
# ---------------------------------------------------------------------------
def compute_diagnosis(results, alias_stats):
    d = {}
    # -- A: D1 stimuli _|_ obs -------------------------------------------
    corrs = []
    for seed in SEEDS:
        for e in results[seed]["A-STD"]["episodes"]:
            corrs.extend(abs(v) for v in e["stim_beacon_corr"].values())
    d["D1_max_abs_stim_beacon_corr_A_STD"] = round(max(corrs), 4)
    d["D1_mean_abs_stim_beacon_corr_A_STD"] = round(_mean(corrs), 4)
    # -- A: D2 gains at floor --------------------------------------------
    floor_frac = []
    for seed in SEEDS:
        for e in results[seed]["A-STD"]["episodes"]:
            g = e["gains_end"]
            floor_frac.append(
                sum(1 for v in g.values() if v <= 0.011) / len(g))
    d["D2_gain_floor_frac_A_STD"] = round(_mean(floor_frac), 4)
    # -- A: D3 dense lift --------------------------------------------------
    d["D3_dense_minus_std_A"] = round(
        _mean(arm_means(results, "A-DENSE").values())
        - _mean(arm_means(results, "A-STD").values()), 4)
    # -- A: D4 near: post-goal vs pre-goal ---------------------------------
    d["D4_post_minus_pre_goal_eps_A_NEAR"] = _post_pre_gap(results, "A-NEAR")
    # -- B: D5 sparse goal rate -------------------------------------------
    d["D5_p_goal_B_STD"] = round(p_goal(results, "B-STD"), 4)
    # -- B: D6 dense lift --------------------------------------------------
    d["D6_dense_minus_std_B"] = round(
        _mean(arm_means(results, "B-DENSE").values())
        - _mean(arm_means(results, "B-STD").values()), 4)
    # -- B: D7 goal-trial registration -------------------------------------
    d["D7_goal_trial_rerr"] = _goal_trial_rerr(results, "B")
    # -- B: D8 aliasing ----------------------------------------------------
    d["D8_aliasing_rate"] = alias_stats["aliasing_rate"]
    d["D8_obs_keys_multi_pos"] = alias_stats["n_multi"]
    d["D8_n_contexts_B_STD"] = _n_contexts_check(results, "B-STD")
    # -- C: D9 flat stimuli + floor + tie-break ----------------------------
    tie, stimstd, gfloor = [], [], []
    for seed in SEEDS:
        for e in results[seed]["C-STD"]["episodes"]:
            tie.append(e["tie_break_frac"])
            stimstd.append(e["stimuli_std"])
            g = e["gains_end"]
            gfloor.append(sum(1 for v in g.values() if v <= 0.011) / len(g))
    d["D9_tie_break_frac_C_STD"] = round(_mean(tie), 4)
    d["D9_stimuli_std_C_STD"] = round(_mean(stimstd), 5)
    d["D9_gain_floor_frac_C_STD"] = round(_mean(gfloor), 4)
    # -- C: D10 near: floor persists + no post-goal improvement ------------
    gfloor_n = []
    for seed in SEEDS:
        for e in results[seed]["C-NEAR"]["episodes"]:
            g = e["gains_end"]
            gfloor_n.append(sum(1 for v in g.values() if v <= 0.011) / len(g))
    d["D10_gain_floor_frac_C_NEAR"] = round(_mean(gfloor_n), 4)
    d["D10_post_minus_pre_goal_eps_C_NEAR"] = _post_pre_gap(results, "C-NEAR")
    # -- C: D11 dense lift --------------------------------------------------
    d["D11_dense_minus_std_C"] = round(
        _mean(arm_means(results, "C-DENSE").values())
        - _mean(arm_means(results, "C-STD").values()), 4)
    # -- DEMO arms: does demonstrated reward transfer? ----------------------
    for arch in ("A", "B", "C"):
        std_m = _mean(arm_means(results, f"{arch}-STD").values())
        demo_m = _mean(arm_means(results, f"{arch}-DEMO").values())
        d[f"DEMO_canonical_gain_{arch}"] = round(demo_m - std_m, 4)
    # -- baselines -----------------------------------------------------------
    for arm in ("RANDOM", "SYMBOLIC", "A-STD", "B-STD", "C-STD"):
        ms = arm_means(results, arm)
        d[f"mean_return_{arm}"] = round(_mean(ms.values()), 4)
        d[f"p_goal_{arm}"] = round(p_goal(results, arm), 4)
    return d


def _post_pre_gap(results, arm):
    """Mean(post-first-goal episodes) - mean(pre-first-goal episodes),
    averaged over seeds that have at least one goal and at least one
    episode on each side."""
    gaps = []
    for seed in SEEDS:
        eps = results[seed][arm]["episodes"]
        first = next((i for i, e in enumerate(eps) if e["goal"]), None)
        if first is None or first == 0 or first == len(eps) - 1:
            continue
        pre = _mean(e["return"] for e in eps[:first])
        post = _mean(e["return"] for e in eps[first + 1:])
        gaps.append(post - pre)
    return round(_mean(gaps), 4) if gaps else None


def _goal_trial_rerr(results, arch):
    """First-half vs second-half mean|rerr| on goal trials (all
    goal probes across STD+DENSE+NEAR arms), plus rhat movement."""
    probes = []
    for variant in ("STD", "DENSE", "NEAR"):
        for seed in SEEDS:
            probes.extend(results[seed][f"{arch}-{variant}"]["goal_probes"])
    if not probes:
        return {"n_probes": 0}
    rerrs = [abs(p["reward_at_trial"] - p["rhat_at_trial"]) for p in probes]
    half = max(1, len(rerrs) // 2)
    move = [abs(p["reward_at_trial"] - p["rhat_at_end"])
            for p in probes]
    return {
        "n_probes": len(probes),
        "mean_abs_rerr_first_half": round(_mean(rerrs[:half]), 4),
        "mean_abs_rerr_second_half": round(_mean(rerrs[half:]), 4),
        "mean_abs_rerr_at_trial": round(_mean(rerrs), 4),
        "mean_abs_rerr_at_end": round(_mean(move), 4),
    }


def _n_contexts_check(results, arm):
    vals = set()
    for seed in SEEDS:
        for e in results[seed][arm]["episodes"]:
            vals.add(e.get("n_contexts"))
    return sorted(v for v in vals if v is not None)


def compute_aliasing(alias_collect):
    """D8: obs-key -> positions -> BFS-optimal actions."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for key, pos, opt in alias_collect:
        by_key[key].append((pos, opt))
    multi = {k: v for k, v in by_key.items() if len({p for p, _ in v}) >= 2}
    disagree = 0
    for k, v in multi.items():
        acts = {a for _, a in v if a is not None}
        if len(acts) > 1:
            disagree += 1
    return {
        "n_obs_keys": len(by_key),
        "n_multi": len(multi),
        "n_disagree": disagree,
        "aliasing_rate": round(disagree / max(1, len(multi)), 4),
        "total_ticks": len(alias_collect),
    }


def failure_verdicts(d):
    """Apply the preregistered failure-class rules. Multiple may fire."""
    v = {}
    # A
    fa = []
    rep_evidence = []
    if d["D1_max_abs_stim_beacon_corr_A_STD"] < 0.10:
        rep_evidence.append("stimuli_perp_obs")
    if d["D2_gain_floor_frac_A_STD"] > 0.75:
        rep_evidence.append("gains_at_floor")
    if d["D3_dense_minus_std_A"] <= 0.5:
        rep_evidence.append("dense_no_lift")
    if rep_evidence:
        fa.append("REPRESENTATION:" + "+".join(rep_evidence))
    if (d["D4_post_minus_pre_goal_eps_A_NEAR"] is not None
            and d["D4_post_minus_pre_goal_eps_A_NEAR"] <= 0.2):
        fa.append("CREDIT_ASSIGNMENT:no_post_goal_improvement")
    v["A"] = fa
    # B
    fb = []
    if d["D5_p_goal_B_STD"] < 0.20 and d["D6_dense_minus_std_B"] > 1.0:
        fb.append("EXPLORATION:sparse_not_found_dense_learnable")
    g7 = d["D7_goal_trial_rerr"]
    if g7.get("n_probes", 0) == 0:
        fb.append("CREDIT_ASSIGNMENT:UNTESTABLE_no_goal_trials")
    elif g7["mean_abs_rerr_second_half"] >= g7["mean_abs_rerr_first_half"]:
        fb.append("CREDIT_ASSIGNMENT:rerr_not_declining")
    if d["D8_aliasing_rate"] > 0.30:
        fb.append("REPRESENTATION:obs_aliasing")
    v["B"] = fb
    # C
    fc = []
    if (d["D9_stimuli_std_C_STD"] < 0.05
            and d["D9_gain_floor_frac_C_STD"] > 0.75
            and d["D9_tie_break_frac_C_STD"] > 0.90):
        fc.append("CREDIT_ASSIGNMENT:habituation_decay(NR-A-006)")
    if (d["D10_gain_floor_frac_C_NEAR"] > 0.75
            and (d["D10_post_minus_pre_goal_eps_C_NEAR"] is None
                 or d["D10_post_minus_pre_goal_eps_C_NEAR"] <= 0.2)):
        fc.append("CREDIT_ASSIGNMENT:floor_persists_despite_reward")
    if d["D11_dense_minus_std_C"] > 1.0:
        fc.append("EXPLORATION:sparse_habituation_loop_dense_lifts")
    else:
        fc.append("REPRESENTATION:dense_no_lift")
    v["C"] = fc
    return v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    em = Emitter()
    lab = os.path.dirname(_HERE)

    # G0: canonical pomaze.py untouched
    pom_path = os.path.join(_ENVS, "pomaze.py")
    with open(pom_path, "rb") as f:
        pom_sha = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(_HERE, "preregistration_POMAZE_DIAG.json")) as f:
        prereg = json.load(f)
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_POMAZE_DIAG.json",
        "prereg_sha256": hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()).hexdigest(),
        "experiment_id": EXPERIMENT_ID,
        "type": prereg["type"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "pomaze_py_sha256": pom_sha,
        "diagnostic_hypotheses": prereg["diagnostic_hypotheses"],
        "failure_classes": prereg["failure_classes"],
    })
    for env_name, cls in (("pomaze", POMaze),
                          ("pomaze_dense", POMazeDense),
                          ("pomaze_near", POMazeNear)):
        mod_file = {"pomaze": "pomaze.py",
                    "pomaze_dense": "pomaze_dense.py",
                    "pomaze_near": "pomaze_near.py"}[env_name]
        with open(os.path.join(_ENVS, mod_file), "rb") as f:
            mod_sha = hashlib.sha256(f.read()).hexdigest()
        em.emit("env_config", {
            "env": env_name, "class": cls.__name__,
            "module_sha256": mod_sha,
        })

    results = {seed: {} for seed in SEEDS}
    alias_collect = []
    for arm in ARMS:
        for seed in SEEDS:
            r = run_arm(arm, seed, alias_collect)
            results[seed][arm] = r
            ms = _mean(e["return"] for e in r["episodes"])
            pg = sum(1 for e in r["episodes"] if e["goal"])
            em.emit("seed_result", {
                "arm": arm, "seed": seed,
                "n_episodes": len(r["episodes"]),
                "mean_return": round(ms, 4),
                "p_goal": round(pg / max(1, len(r["episodes"])), 4),
                "episodes": r["episodes"],
                "n_goal_probes": len(r.get("goal_probes", [])),
                "goal_probes": [
                    {k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in p.items() if k != "obs_vec"}
                    for p in r.get("goal_probes", [])],
            })
            tag = f"{arm:10s} seed={seed}: mean={ms:+.4f} p_goal={pg}/{len(r['episodes'])}"
            print("  " + tag, flush=True)

    alias_stats = compute_aliasing(alias_collect)
    em.emit("aliasing", alias_stats)
    print(f"  aliasing: rate={alias_stats['aliasing_rate']} "
          f"(n_multi={alias_stats['n_multi']})", flush=True)

    d = compute_diagnosis(results, alias_stats)
    em.emit("diagnosis", d)
    v = failure_verdicts(d)
    em.emit("verdict", {"per_arch_failure_classes": v})
    print("  verdict: " + json.dumps(v), flush=True)

    # G2 determinism: recompute (B-STD, 91001)
    r1 = _mean(e["return"] for e in results[91001]["B-STD"]["episodes"])
    r2res = run_arm("B-STD", 91001, [])
    r2 = _mean(e["return"] for e in r2res["episodes"])
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "arm": "B-STD", "seed": 91001, "first": r1, "rerun": r2,
        "match_1e9": bool(det_ok)})
    print(f"  G2 determinism: {r1:.6f} vs {r2:.6f} -> "
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
