"""EXP-FP-C-BUILD-AND-BEAT — battery driver (preregistered).

Runs arch-A (current, R1 default), arch-B (intact), arch-C (the hybrid),
and three C ablation arms through ONE driver loop with identical env
protocol and identical paired seeds.

Frozen by experiments/preregistration_ARCH_C.json. No tuning, no
retrospective changes: the driver reads the frozen config below and the
prereg JSON only for documentation.

Receipts: hash-chained ndjson
  prototypes/architecture-c/receipts/EXP-FP-C-BUILD-AND-BEAT.ndjson
(record kinds: preregistration, env_config, seed_result, env_verdict,
 overall_verdict; each carries prev_hash + receipt_hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_C = os.path.dirname(_HERE)
_ARCH_A = os.path.join(os.path.dirname(_ARCH_C), "architecture-a")
_ARCH_B = os.path.join(os.path.dirname(_ARCH_C), "architecture-b")
_FLESH = os.path.dirname(os.path.dirname(_ARCH_C))
_FLESH_EXP = os.path.join(_FLESH, "experiments")
_FLESH_ENVS = os.path.join(_FLESH_EXP, "envs")
for _p in (_ARCH_C, _ARCH_A, _ARCH_B, _FLESH_EXP, _FLESH_ENVS,
           os.path.join(_ARCH_A, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_agent import ArchC, CAdapter  # noqa: E402
from tick import WorkspaceTick  # noqa: E402
from attention_cue import CueIndexedArbitrator  # noqa: E402
from k5_multitask_generalization import neutral_specialists  # noqa: E402
from agent import ArchB  # noqa: E402
from env_interface import derive_seed  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402
from compositional_rule import CompositionalRule  # noqa: E402
from pomaze import POMaze  # noqa: E402
from delayed_reward import DelayedReward  # noqa: E402
from cue_delayed_reward import CueDelayedReward  # noqa: E402

EXPERIMENT_ID = "EXP-FP-C-BUILD-AND-BEAT"
SEEDS = (80101, 80102, 80103, 80104)
THETA = 0.45
GAIN_LR = 0.15

ENVS = {
    "changing_rule": {
        "cls": ChangingRule, "channels": ["a0", "a1"],
        "c2a": {"a0": 0, "a1": 1}, "episodes": 12,
        "context_fn": lambda obs: int(obs["cue"]), "margin": 20.0},
    "compositional_rule": {
        "cls": CompositionalRule, "channels": ["a0", "a1"],
        "c2a": {"a0": 0, "a1": 1}, "episodes": 12,
        "context_fn": lambda obs: (int(obs["cue_a"]), int(obs["cue_b"])),
        "margin": 20.0},
    "pomaze": {
        "cls": POMaze,
        "channels": ["north", "south", "east", "west"],
        "c2a": {"north": 0, "south": 1, "east": 2, "west": 3},
        "episodes": 15, "context_fn": None, "margin": 0.5},
    "delayed_reward": {
        "cls": DelayedReward,
        "channels": ["branch_a", "branch_b", "forward", "stay"],
        "c2a": {"branch_a": 0, "branch_b": 1, "forward": 2, "stay": 3},
        "episodes": 20, "context_fn": None, "margin": 0.15},
    "cue_delayed_reward": {
        "cls": CueDelayedReward,
        "channels": ["branch_a", "branch_b", "forward", "stay"],
        "c2a": {"branch_a": 0, "branch_b": 1, "forward": 2, "stay": 3},
        "episodes": 20,
        "context_fn": lambda obs: int(obs["cue"]), "margin": 0.15},
}

ARMS = ["A", "B", "C"]
ABLATION_ARMS = ["C-frozen-predictor", "C-no-memory", "C-frozen-gains"]
ABLATION_ENVS = ("changing_rule", "pomaze")


# ---------------------------------------------------------------------------
# agent constructors (one instance per run; cross-episode learning persists)
# ---------------------------------------------------------------------------
def make_A(env_cfg, env_obj, agent_seed):
    channels = env_cfg["channels"]
    context_fn = env_cfg["context_fn"]
    arb_cls = CueIndexedArbitrator if context_fn is not None else None
    tick = WorkspaceTick(
        channels, neutral_specialists(channels, agent_seed),
        capacity=len(channels), frozen_gains=False, gain_lr=GAIN_LR,
        ignition_kwargs={"theta": THETA},
        arbitrator_cls=arb_cls, context_fn=context_fn)
    return ("A", tick)


def make_B(env_cfg, env_obj, agent_seed):
    space = env_obj.observation_space()
    b = ArchB(space, len(env_cfg["channels"]),
              env_name=env_obj.NAME,
              action_mode="active_inference", affect="none",
              seed=agent_seed)
    return ("B", b)


def make_C(env_cfg, env_obj, agent_seed, variant="C"):
    space = env_obj.observation_space()
    cfg = {
        "channels": env_cfg["channels"],
        "channel_to_action": env_cfg["c2a"],
        "observation_space": space,
        "env_name": env_obj.NAME,
        "context_fn": env_cfg["context_fn"],
        "gain_lr": GAIN_LR, "theta": THETA, "kappa": 0.1,
        "agent_seed": agent_seed,
        "predictor_kwargs": {"eta0": 0.005, "etaD": 0.002, "eta_r": 0.05,
                             "etaR": 0.10, "precision_kind": "estimated",
                             "max_contexts": 32, "precision_window": 50},
        "memory_enabled": variant != "C-no-memory",
        "frozen_predictor": variant == "C-frozen-predictor",
        "frozen_gains": variant == "C-frozen-gains",
    }
    return (variant, ArchC(cfg))


# ---------------------------------------------------------------------------
# one run: n_episodes through the shared driver; returns per-episode returns
# ---------------------------------------------------------------------------
def run_arm(arm, env_name, primary_seed):
    env_cfg = ENVS[env_name]
    env = env_cfg["cls"]()
    agent_seed = derive_seed(primary_seed, 0, "agent")
    if arm == "A":
        _name, agent = make_A(env_cfg, env, agent_seed)
    elif arm == "B":
        _name, agent = make_B(env_cfg, env, agent_seed)
    else:
        _name, agent = make_C(env_cfg, env, agent_seed, variant=arm)

    ep_returns = []
    sub_ign = 0
    n_episodes = env_cfg["episodes"]
    for run_index in range(n_episodes):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        if arm == "B":
            ep_returns.append(_run_episode_B(env, agent, ep_seed, run_index))
        else:
            ret, si = _run_episode_tick(env, agent, ep_seed, env_cfg, arm)
            ep_returns.append(ret)
            sub_ign += si
    mean_ret = sum(ep_returns) / len(ep_returns)
    diag = {}
    if arm != "B":
        st = agent.stats()
        diag = {"final_gains": {k: round(v, 4) for k, v in
                                st.get("gains", {}).items()},
                "broadcast_items": st.get("broadcast_items"),
                "sub_ignition_actions": sub_ign}
        if arm != "A":
            diag["memory_size"] = st.get("memory_size")
            diag["n_contexts"] = st.get("n_contexts")
    return {"mean_return": mean_ret, "ep_returns": [round(r, 4) for r in
                                                   ep_returns],
            "diagnostics": diag}


def _run_episode_tick(env, agent, ep_seed, env_cfg, arm):
    """Shared driver for A and C: CAdapter protocol; explicit resets.
    Returns (episode_return, n_sub_ignition_actions)."""
    adapter = CAdapter(env, env_cfg["c2a"])
    obs = adapter.reset_episode(ep_seed)
    if arm != "A":
        agent.reset_episode(obs)
    total, sub_ign = 0.0, 0
    while True:
        if arm == "A":
            trace = agent.step(obs, adapter)
        else:
            trace = agent.step(adapter)
        total += trace["reward"]
        if trace.get("action_selection", {}).get("path") == \
                "sub_ignition_explore":
            sub_ign += 1
        _obs_next, _r, done, _info = adapter.last_transition()
        if done:
            break
        obs = adapter.observe()
    return total, sub_ign


def _run_episode_B(env, agent, ep_seed, run_index):
    """B via the Agent ABC (harness convention)."""
    obs = env.reset(ep_seed)
    agent.reset(derive_seed(ep_seed, run_index, "agent"),
                env.action_space())
    total, done = 0.0, False
    while not done:
        a = agent.act(obs)
        obs2, reward, done, info = env.step(a)
        agent.update(obs, a, reward, done, info)
        total += reward
        obs = obs2
    return total


# ---------------------------------------------------------------------------
# hash-chained receipt emission (K10 convention)
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


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    em = Emitter()
    with open(os.path.join(_FLESH_EXP,
                           "preregistration_ARCH_C.json")) as f:
        prereg = json.load(f)
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_ARCH_C.json",
        "prereg_sha256": hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()).hexdigest(),
        "hypothesis": prereg["hypothesis"],
        "null": prereg["null"],
        "metric": prereg["metric"],
        "seeds": list(SEEDS),
        "win_rule_per_env": prereg["win_rule_per_env"],
        "overall_verdict": prereg["overall_verdict"],
        "margins": {e: c["margin"] for e, c in ENVS.items()},
        "frozen_config": prereg["frozen_config"],
    })

    results = {}  # results[env][arm][seed] = mean_return
    for env_name, env_cfg in ENVS.items():
        em.emit("env_config", {
            "env": env_name,
            "episodes": env_cfg["episodes"],
            "channels": env_cfg["channels"],
            "margin": env_cfg["margin"],
            "context": ("cue-indexed" if env_cfg["context_fn"] is not None
                        else "default"),
        })
        results[env_name] = {}
        arms = list(ARMS)
        if env_name in ABLATION_ENVS:
            arms += ABLATION_ARMS
        for arm in arms:
            results[env_name][arm] = {}
            for seed in SEEDS:
                r = run_arm(arm, env_name, seed)
                results[env_name][arm][seed] = r["mean_return"]
                em.emit("seed_result", {
                    "env": env_name, "arm": arm, "seed": seed,
                    "mean_return": round(r["mean_return"], 4),
                    "ep_returns": r["ep_returns"],
                    "diagnostics": r["diagnostics"],
                })
                print(f"  {env_name:20s} {arm:18s} seed={seed}: "
                      f"mean_return={r['mean_return']:.4f}", flush=True)
        # per-env verdicts vs A and vs B
        env_verdicts = {}
        for other in ("A", "B"):
            wins = 0
            deltas = {}
            for seed in SEEDS:
                d = (results[env_name]["C"][seed]
                     - results[env_name][other][seed])
                deltas[str(seed)] = round(d, 4)
                wins += 1 if d > 0 else 0
            mean_d = sum(deltas.values()) / len(deltas)
            win = bool(mean_d > env_cfg["margin"] and wins >= 3)
            env_verdicts[f"C_vs_{other}"] = {
                "win": win, "mean_delta": round(mean_d, 4),
                "margin": env_cfg["margin"], "seeds_agreeing": wins,
                "per_seed_delta": deltas}
        em.emit("env_verdict", {"env": env_name, **env_verdicts})
        print(f"  -> {env_name}: C_vs_A win={env_verdicts['C_vs_A']['win']} "
              f"(d={env_verdicts['C_vs_A']['mean_delta']:+.3f}), "
              f"C_vs_B win={env_verdicts['C_vs_B']['win']} "
              f"(d={env_verdicts['C_vs_B']['mean_delta']:+.3f})", flush=True)

    beats_A = sum(1 for e in ENVS
                  if _env_win(results, e, "A")) >= 4
    beats_B = sum(1 for e in ENVS
                  if _env_win(results, e, "B")) >= 4
    overall = bool(beats_A and beats_B)
    em.emit("overall_verdict", {
        "C_BEATS_A": bool(beats_A), "C_BEATS_B": bool(beats_B),
        "C_BEATS": overall,
        "env_wins_vs_A": [e for e in ENVS if _env_win(results, e, "A")],
        "env_wins_vs_B": [e for e in ENVS if _env_win(results, e, "B")],
        "rule": "C_BEATS iff wins >= 4/5 envs vs A AND >= 4/5 vs B",
        "interpretation": ("C BEATS A and B by preregistered margins"
                           if overall else
                           "FAIL — C does not beat A and B by the "
                           "preregistered margins; per-env breakdown above "
                           "is the honest result"),
    })

    # G0c determinism spot-check: rerun (changing_rule, 80101, C).
    r1 = results["changing_rule"]["C"][80101]
    r2 = run_arm("C", "changing_rule", 80101)["mean_return"]
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "env": "changing_rule", "arm": "C", "seed": 80101,
        "first": r1, "rerun": r2, "match_1e9": bool(det_ok)})
    print(f"G0c determinism: {r1:.6f} vs {r2:.6f} -> "
          f"{'MATCH' if det_ok else 'MISMATCH'}", flush=True)

    ok, msg = em.verify()
    print(f"chain verify: {ok} ({msg})", flush=True)

    outdir = os.path.join(_ARCH_C, "receipts")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{EXPERIMENT_ID}.ndjson")
    with open(out, "w") as f:
        for rec in em.records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"receipt: {out} ({len(em.records)} records, "
          f"{time.time()-t0:.1f}s)", flush=True)
    # per-env JSON summaries (derived, for the handoff docs)
    summ = os.path.join(outdir, f"{EXPERIMENT_ID}.summary.json")
    with open(summ, "w") as f:
        json.dump({"experiment_id": EXPERIMENT_ID,
                   "results": {e: {a: {str(s): round(v, 4)
                                        for s, v in sd.items()}
                                   for a, sd in ad.items()}
                               for e, ad in results.items()},
                   "overall": {"C_BEATS_A": bool(beats_A),
                               "C_BEATS_B": bool(beats_B),
                               "C_BEATS": overall}},
                  f, indent=2, sort_keys=True)
    print(f"summary: {summ}", flush=True)


def _env_win(results, env, other):
    m = ENVS[env]["margin"]
    ds = [results[env]["C"][s] - results[env][other][s] for s in SEEDS]
    return sum(ds) / len(ds) > m and sum(1 for d in ds if d > 0) >= 3


if __name__ == "__main__":
    main()
