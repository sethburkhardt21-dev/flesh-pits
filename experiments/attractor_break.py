"""EXP-FP-0011-ATTRACTOR-BREAK — causal test of the EXP-FP-0010 bump-attractor.

CAUSAL (not a characterization). Preregistered:
  experiments/preregistration_ATTRACTOR_BREAK.json  (sealed BEFORE this ran)

Question: does breaking arch-C's wall-bump attractor (0010 mechanism:
gains floored at 0.01 -> arbitration = argmax of habituated raw bids ->
raw bids respond only to stimulus changes -> frozen winner -> bump
attractor) require the self-reinforcing habituation loop, the floored
gain loop, or neither?

Arms (6 fresh primary seeds 92001..92006, paired):
  C-BASE            : unmodified ArchC (reference)
  C-REPEAT-PENALTY  : ArchC + anti-habituation repeat penalty in the bid
  C-GAIN-LIFT       : ArchC + gain floor 0.01 -> 0.5

Envs: canonical pomaze (primary, 15 eps/seed); changing_rule and
compositional_rule (control, 12 eps/seed — healthy tasks where C~=A).

Frozen gates: G1 stuck-frac drops by >=0.3 AND mean return improves by
>=1.0 (seed-averaged, paired, canonical pomaze) for an intervention to
count as breaking the attractor. G2 control: changing_rule seed-mean >=
0.7 * C-BASE seed-mean.

Receipts: hash-chained ndjson receipts/EXP-FP-0011-ATTRACTOR-BREAK.ndjson.
Canonical c_agent.py / attention.py / attention_cue.py / tick.py and the
three env modules untouched (sha256 recorded, G0). Nothing pushed (G6).

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
_ARCH_C = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-c")
_ARCH_A = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-a")
_ARCH_B = os.path.join(os.path.dirname(_HERE), "prototypes", "architecture-b")
_ENVS = os.path.join(_HERE, "envs")
for _p in (_ARCH_C, _ARCH_A, _ARCH_B, _HERE, _ENVS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_agent import ArchC, CAdapter  # noqa: E402
from c_attractor_break import (ArchCRepeatPenalty, ArchCGainLift,  # noqa: E402
                               ArchCAttractorBreak)
from env_interface import derive_seed  # noqa: E402
from pomaze import POMaze  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402
from compositional_rule import CompositionalRule  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0011-ATTRACTOR-BREAK"
SEEDS = (92001, 92002, 92003, 92004, 92005, 92006)
THETA = 0.45
GAIN_LR = 0.15

ENVS = {
    "pomaze": {
        "cls": POMaze, "channels": ["north", "south", "east", "west"],
        "c2a": {"north": 0, "south": 1, "east": 2, "west": 3},
        "episodes": 15, "context_fn": None, "role": "primary"},
    "changing_rule": {
        "cls": ChangingRule, "channels": ["a0", "a1"],
        "c2a": {"a0": 0, "a1": 1}, "episodes": 12,
        "context_fn": lambda obs: int(obs["cue"]), "role": "control"},
    "compositional_rule": {
        "cls": CompositionalRule, "channels": ["a0", "a1"],
        "c2a": {"a0": 0, "a1": 1}, "episodes": 12,
        "context_fn": lambda obs: (int(obs["cue_a"]), int(obs["cue_b"])),
        "role": "control"},
}

ARMS = ["C-BASE", "C-REPEAT-PENALTY", "C-GAIN-LIFT"]
ARM_CLS = {
    "C-BASE": ArchC,
    "C-REPEAT-PENALTY": ArchCRepeatPenalty,
    "C-GAIN-LIFT": ArchCGainLift,
}


# ---------------------------------------------------------------------------
# agent constructors — byte-identical config to 0010 / build-and-beat C
# ---------------------------------------------------------------------------
def make_agent(arm, env_name, env_obj, agent_seed):
    cfg_env = ENVS[env_name]
    cfg = {
        "channels": cfg_env["channels"],
        "channel_to_action": cfg_env["c2a"],
        "observation_space": env_obj.observation_space(),
        "env_name": env_obj.NAME, "context_fn": cfg_env["context_fn"],
        "gain_lr": GAIN_LR, "theta": THETA, "kappa": 0.1,
        "agent_seed": agent_seed,
        "predictor_kwargs": {"eta0": 0.005, "etaD": 0.002, "eta_r": 0.05,
                             "etaR": 0.10, "precision_kind": "estimated",
                             "max_contexts": 32, "precision_window": 50},
        "memory_enabled": True, "frozen_predictor": False,
        "frozen_gains": False,
    }
    return ARM_CLS[arm](cfg)


def gains_of(agent):
    return dict(agent.tick.arbitrator.gains)


# ---------------------------------------------------------------------------
# episode runners (0010 / battery conventions)
# ---------------------------------------------------------------------------
def run_pomaze_episode(agent, env, ep_seed, c2a):
    adapter = CAdapter(env, c2a)
    obs = adapter.reset_episode(ep_seed)
    agent.reset_episode(obs)
    g0 = gains_of(agent)
    total, stuck, goal = 0.0, 0, False
    tick_n = 0
    margins = []
    while True:
        pos_before = env._pos
        trace = agent.step(adapter)
        total += trace["reward"]
        tick_n += 1
        margins.append(trace["arbitration"]["margin"])
        if env._pos == pos_before:
            stuck += 1
        _obs_next, _r, done, info = adapter.last_transition()
        if info.get("goal_reached") and not goal:
            goal = True
        if done:
            break
        obs = adapter.observe()
    g1 = gains_of(agent)
    tie_frac = sum(1 for m in margins if m < 1e-9) / max(1, len(margins))
    return {"return": round(total, 4), "goal": goal,
            "stuck_frac": round(stuck / max(1, tick_n), 4),
            "ticks": tick_n,
            "tie_break_frac": round(tie_frac, 4),
            "margin_mean": round(sum(margins) / max(1, len(margins)), 6),
            "gains_start": {k: round(v, 4) for k, v in g0.items()},
            "gains_end": {k: round(v, 4) for k, v in g1.items()},
            "floor_frac": round(
                sum(1 for v in g1.values() if v <= 0.011) / len(g1), 4),
            "gain_mean_end": round(sum(g1.values()) / len(g1), 4)}


def run_rule_episode(agent, env, ep_seed, c2a):
    adapter = CAdapter(env, c2a)
    obs = adapter.reset_episode(ep_seed)
    agent.reset_episode(obs)
    total = 0.0
    while True:
        trace = agent.step(adapter)
        total += trace["reward"]
        _obs_next, _r, done, _info = adapter.last_transition()
        if done:
            break
        obs = adapter.observe()
    return {"return": round(total, 4)}


def run_arm(arm, env_name, primary_seed):
    env_cfg = ENVS[env_name]
    env = env_cfg["cls"]()
    agent = make_agent(arm, env_name, env,
                       derive_seed(primary_seed, 0, "agent"))
    eps = []
    for run_index in range(env_cfg["episodes"]):
        ep_seed = derive_seed(primary_seed, run_index, "env")
        if env_name == "pomaze":
            eps.append(run_pomaze_episode(agent, env, ep_seed,
                                          env_cfg["c2a"]))
        else:
            eps.append(run_rule_episode(agent, env, ep_seed,
                                        env_cfg["c2a"]))
    return agent, eps


# ---------------------------------------------------------------------------
# hash-chained receipt emission (0010 convention)
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


def sha_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    em = Emitter()
    lab = os.path.dirname(_HERE)

    with open(os.path.join(_HERE,
                           "preregistration_ATTRACTOR_BREAK.json")) as f:
        prereg = json.load(f)
    em.emit("preregistration", {
        "prereg_file": "experiments/preregistration_ATTRACTOR_BREAK.json",
        "prereg_sha256": sha_of(
            os.path.join(_HERE, "preregistration_ATTRACTOR_BREAK.json")),
        "experiment_id": EXPERIMENT_ID,
        "type": prereg["type"], "seeds": list(SEEDS),
        "arms": list(ARMS),
        "hypothesis": prereg["hypothesis"], "null": prereg["null"],
        "verdict_mapping": prereg["verdict_mapping"],
    })
    # G0: canonical files untouched
    for label, path in (
            ("c_agent", os.path.join(_ARCH_C, "c_agent.py")),
            ("attention", os.path.join(_ARCH_A, "attention.py")),
            ("attention_cue", os.path.join(_ARCH_A, "attention_cue.py")),
            ("tick", os.path.join(_ARCH_A, "tick.py")),
            ("pomaze", os.path.join(_ENVS, "pomaze.py")),
            ("changing_rule", os.path.join(_ENVS, "changing_rule.py")),
            ("compositional_rule",
             os.path.join(_ENVS, "compositional_rule.py"))):
        em.emit("canonical_sha", {"module": label, "sha256": sha_of(path)})
    # new additive files
    for label, path in (
            ("c_attractor_break",
             os.path.join(_ARCH_C, "c_attractor_break.py")),
            ("driver", os.path.join(_HERE, "attractor_break.py"))):
        em.emit("additive_sha", {"module": label, "sha256": sha_of(path)})

    results = {}  # results[env][arm][seed] = {"episodes": [...]}
    for env_name, env_cfg in ENVS.items():
        em.emit("env_config", {
            "env": env_name, "role": env_cfg["role"],
            "episodes": env_cfg["episodes"],
            "channels": env_cfg["channels"]})
        results[env_name] = {}
        for arm in ARMS:
            results[env_name][arm] = {}
            for seed in SEEDS:
                _agent, eps = run_arm(arm, env_name, seed)
                results[env_name][arm][seed] = eps
                ms = _mean(e["return"] for e in eps)
                line = (f"  {env_name:18s} {arm:16s} seed={seed}: "
                        f"mean_return={ms:+.4f}")
                if env_name == "pomaze":
                    sf = _mean(e["stuck_frac"] for e in eps)
                    line += f" mean_stuck={sf:.4f}"
                print(line, flush=True)
                payload = {"env": env_name, "arm": arm, "seed": seed,
                           "n_episodes": len(eps),
                           "mean_return": round(ms, 4),
                           "episodes": eps}
                if env_name == "pomaze":
                    payload["mean_stuck_frac"] = round(
                        _mean(e["stuck_frac"] for e in eps), 4)
                em.emit("seed_result", payload)

    # G1 decision metrics (pomaze, seed-averaged, paired)
    verdict = {}
    for arm in ("C-REPEAT-PENALTY", "C-GAIN-LIFT"):
        ds, dr = [], []
        for seed in SEEDS:
            b = _mean(e["return"] for e in results["pomaze"]["C-BASE"][seed])
            i = _mean(e["return"] for e in results["pomaze"][arm][seed])
            sb = _mean(e["stuck_frac"]
                       for e in results["pomaze"]["C-BASE"][seed])
            si = _mean(e["stuck_frac"]
                       for e in results["pomaze"][arm][seed])
            dr.append(i - b)
            ds.append(sb - si)
        verdict[arm] = {
            "delta_stuck_mean": round(_mean(ds), 4),
            "delta_return_mean": round(_mean(dr), 4),
            "G1_break": bool(_mean(ds) >= 0.3 and _mean(dr) >= 1.0)}
    # G2 control (changing_rule): intervention not destroyed
    base_cr = _mean(_mean(e["return"]
                          for e in results["changing_rule"]["C-BASE"][s])
                    for s in SEEDS)
    for arm in ("C-REPEAT-PENALTY", "C-GAIN-LIFT"):
        m = _mean(_mean(e["return"]
                        for e in results["changing_rule"][arm][s])
                  for s in SEEDS)
        verdict[arm]["changing_rule_mean"] = round(m, 4)
        verdict[arm]["G2_control_ok"] = bool(m >= 0.7 * base_cr)
    verdict["C-BASE_changing_rule_mean"] = round(base_cr, 4)
    # descriptive: compositional_rule
    for arm in ARMS:
        m = _mean(_mean(e["return"]
                        for e in results["compositional_rule"][arm][s])
                  for s in SEEDS)
        verdict[f"{arm}_compositional_rule_mean"] = round(m, 4)
    em.emit("verdict", verdict)
    print("  verdict: " + json.dumps(verdict), flush=True)

    # G4 determinism: recompute (pomaze, C-BASE, 92001)
    r1 = _mean(e["return"]
               for e in results["pomaze"]["C-BASE"][92001])
    _a2, eps2 = run_arm("C-BASE", "pomaze", 92001)
    r2 = _mean(e["return"] for e in eps2)
    det_ok = abs(r1 - r2) < 1e-9
    em.emit("determinism_check", {
        "env": "pomaze", "arm": "C-BASE", "seed": 92001,
        "first": r1, "rerun": r2, "match_1e9": bool(det_ok)})
    print(f"  G4 determinism: {r1:.6f} vs {r2:.6f} -> "
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
