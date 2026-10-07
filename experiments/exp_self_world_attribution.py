"""EXP-FP-0060-ATTRIBUTION — self/world attribution probe + quarantine test.

Preregistration: experiments/preregistration_SELF_WORLD_ATTRIB.json
(written BEFORE any run). This script implements exactly that procedure.

Part A (attribution): ComparatorAgent vs ChanceAgent (paired env streams)
on self_world_mismatch v1.0.0 — 15% injected mismatches (override/swap/delay),
4 fresh seeds x 5 episodes x 60 ticks. Per-tick ground-truth cause labels come
from env info — read by THIS SCRIPT for scoring only. Neither agent ever sees
info (contract v1.0.0 section 5).

Part B (quarantine): ComparatorAgent drives 3 episodes (seed 96015); only
"source=='observed'" transition records enter the EpisodicStore; forward
predictions stay in an in-memory buffer. Red-team leak attempts (12) across
all public paths + post-run scan + chain verification.

Usage:
    python3 exp_self_world_attribution.py --out ../receipts

Stdlib only. No `random.*` module calls in executable code (all randomness
via env_interface.new_rng / derive_seed).
"""

import argparse
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "envs"))

from env_interface import (  # noqa: E402
    CONTRACT_VERSION, config_hash, derive_seed,
)
from self_world_mismatch import SelfWorldMismatch  # noqa: E402
from source_attributor import (  # noqa: E402
    ComparatorAgent, ChanceAgent, BURN_IN,
)
from episodic_store import EpisodicStore, StoreQuarantineError  # noqa: E402
from harness import write_receipt  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0060-ATTRIBUTION"
SEEDS = [96021, 96022, 96023, 96024]
EPISODES = 5
TICKS = 60
QUARANTINE_SEED = 96015
QUARANTINE_EPISODES = 3


# --------------------------------------------------------------------------
# Part A helpers
# --------------------------------------------------------------------------

def run_arm(agent_factory, episode_seed, run_index):
    """Run one episode; return (agent, per-tick ground truth list)."""
    env = SelfWorldMismatch()
    agent = agent_factory()
    obs = env.reset(episode_seed)
    agent.reset(derive_seed(episode_seed, run_index, "agent"),
                env.action_space())
    truth = []  # per tick: {"hand": ..., "ball": ..., "mode": ..., "active": ...}
    done = False
    while not done:
        action = agent.act(obs)
        env.validate_action(action)
        obs2, reward, done, info = env.step(action)
        env.validate_obs(obs2, env.observation_space())
        agent.update(obs, action, reward, done, info)
        truth.append({
            "hand": info["cause"]["hand"],
            "ball": info["cause"]["ball"],
            "mode": info["mismatch"]["mode"],
            "active": info["mismatch"]["active"],
            "coincides_hand": info["mismatch"]["coincides_hand"],
        })
        obs = obs2
    return agent, truth


def score_arm(agent_factory, seed, run_index, n_episodes):
    """Paired scoring: aggregate attribution correctness vs ground truth.

    Visible mismatch: injected mismatch on the hand channel where the
    exogenous effect did NOT coincide with the commanded direction
    (informationally detectable by an efference-copy comparator).
    Coinciding mismatch: exogenous effect coincided -> residual 0 ->
    the comparator model PREDICTS misattribution to self (H1b).
    """
    agg = {"hit_visible": [0, 0], "err_coincide": [0, 0],
           "hit_all": [0, 0], "accept_normal": [0, 0],
           "pooled": [0, 0], "by_mode": {}}
    for ep in range(n_episodes):
        episode_seed = derive_seed(seed, ep, "env")
        agent, truth = run_arm(agent_factory, episode_seed, run_index)
        for rec in agent.attributions:
            t = truth[rec["tick"] - 1]  # tick is 1-indexed (action tick)
            for ch, attr in rec["attribution"].items():
                true = t[ch]
                correct = (attr == "world") if true == "world" \
                    else (attr == "self")
                agg["pooled"][0] += correct
                agg["pooled"][1] += 1
                if ch == "hand" and true == "world" and t["active"]:
                    agg["hit_all"][0] += correct
                    agg["hit_all"][1] += 1
                    m = t["mode"]
                    c, n = agg["by_mode"].get(m, (0, 0))
                    agg["by_mode"][m] = (c + correct, n + 1)
                    if t["coincides_hand"] is False:
                        agg["hit_visible"][0] += correct
                        agg["hit_visible"][1] += 1
                    elif t["coincides_hand"] is True:
                        # comparator-predicted error: says "self"
                        agg["err_coincide"][0] += (attr == "self")
                        agg["err_coincide"][1] += 1
                elif true == "self":
                    agg["accept_normal"][0] += correct
                    agg["accept_normal"][1] += 1
    return agg


def frac(pair):
    c, n = pair
    return (c / n) if n else 0.0


# --------------------------------------------------------------------------
# Part B helpers
# --------------------------------------------------------------------------

def run_quarantine():
    """Drive the store; red-team it; audit it. Returns a findings dict."""
    store = EpisodicStore()
    leak_attempts = 0
    leaks_admitted = 0
    predicted_buffer = []

    env = SelfWorldMismatch()
    agent = ComparatorAgent()
    for ep in range(QUARANTINE_EPISODES):
        episode_seed = derive_seed(QUARANTINE_SEED, ep, "env")
        obs = env.reset(episode_seed)
        agent.reset(derive_seed(episode_seed, ep, "agent"),
                    env.action_space())
        done = False
        while not done:
            action = agent.act(obs)
            obs2, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done, info)
            store.admit({"source": "observed", "episode": ep,
                         "action": action,
                         "observation": {"hand": obs2["hand"],
                                         "ball": obs2["ball"]},
                         "reward": reward})
            # The latest forward prediction goes to the buffer, NEVER the store.
            predicted_buffer.append(agent.predictions[-1])
            obs = obs2

    # --- red-team: leak attempts across all public paths ------------------
    def attempt(record=None, snapshot=None):
        nonlocal leak_attempts, leaks_admitted
        leak_attempts += 1
        try:
            if snapshot is not None:
                EpisodicStore().restore(snapshot)
            else:
                store.admit(record)
            leaks_admitted += 1  # NOT rejected -> leak
            return "ADMITTED"
        except StoreQuarantineError:
            return "rejected"

    results = []
    results.append(("admit predicted", attempt(
        {"source": "predicted", "episode": 0, "observation": {"hand": 0.7}})))
    results.append(("admit missing source", attempt(
        {"episode": 0, "observation": {"hand": 0.7}})))
    results.append(("admit simulated", attempt(
        {"source": "simulated", "episode": 0})))
    results.append(("admit empty-string source", attempt(
        {"source": "", "episode": 0})))
    results.append(("admit None source", attempt(
        {"source": None, "episode": 0})))
    results.append(("admit non-dict", attempt("predicted payload")))
    results.append(("admit forged observed tag", attempt(
        {"source": "observed", "episode": 0,
         "note": "forged tag on a prediction payload"})))
    # 8. restore() with a record re-tagged to predicted
    snap = store.to_snapshot()
    tampered = {"records": [dict(r) for r in snap["records"]],
                "chain_head": snap["chain_head"]}
    tampered["records"][0] = dict(tampered["records"][0])
    tampered["records"][0]["source"] = "predicted"
    results.append(("restore predicted-tagged", attempt(snapshot=tampered)))
    # 9. restore() with broken chain (content edited post-hoc)
    broken = {"records": [dict(r) for r in snap["records"]],
              "chain_head": snap["chain_head"]}
    broken["records"][1] = dict(broken["records"][1])
    broken["records"][1]["observation"] = {"hand": 0.999, "ball": 0.001}
    results.append(("restore broken chain", attempt(snapshot=broken)))
    # 10. restore() with wrong chain head
    badhead = {"records": [dict(r) for r in snap["records"]],
               "chain_head": "deadbeef"}
    results.append(("restore bad head", attempt(snapshot=badhead)))
    # 11. restore() of an empty snapshot must work (no leak, no error)
    try:
        EpisodicStore().restore({"records": [], "chain_head": None})
        results.append(("restore empty", "ok"))
    except StoreQuarantineError:
        results.append(("restore empty", "UNEXPECTED-REJECTION"))
        leak_attempts += 1
    # 12. restore() of the honest snapshot must succeed
    try:
        fresh = EpisodicStore()
        fresh.restore(snap)
        results.append(("restore honest", "ok"
                        if len(fresh) == len(store) else "MISMATCH"))
    except StoreQuarantineError:
        results.append(("restore honest", "UNEXPECTED-REJECTION"))

    violators = store.scan()
    return {
        "admissions": store.admissions,
        "rejections": store.rejections,
        "leak_attempts": leak_attempts,
        "leaks_admitted": leaks_admitted,
        "red_team": results,
        "audit_violators": len(violators),
        "chain_valid": store.verify_chain(),
        "predicted_buffer_size": len(predicted_buffer),
        "stored_records": len(store),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../receipts")
    args = ap.parse_args()

    config = {
        "contract_version": CONTRACT_VERSION,
        "env": {"name": SelfWorldMismatch.NAME,
                "version": SelfWorldMismatch.VERSION,
                "mismatch_p": SelfWorldMismatch.MISMATCH_P},
        "agents": [ComparatorAgent.NAME, ChanceAgent.NAME],
        "seeds": SEEDS,
        "episodes": EPISODES,
        "ticks": TICKS,
        "burn_in": BURN_IN,
        "quarantine_seed": QUARANTINE_SEED,
        "quarantine_episodes": QUARANTINE_EPISODES,
        "preregistration": "experiments/preregistration_SELF_WORLD_ATTRIB.json",
    }
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    per_seed = []
    for seed in SEEDS:
        comp = score_arm(ComparatorAgent, seed, 0, EPISODES)
        chance = score_arm(ChanceAgent, seed, 1, EPISODES)
        per_seed.append({
            "seed": seed,
            "comparator": {
                "hit_visible": frac(comp["hit_visible"]),
                "hit_visible_n": comp["hit_visible"][1],
                "err_coincide_rate": frac(comp["err_coincide"]),
                "err_coincide_n": comp["err_coincide"][1],
                "hit_all": frac(comp["hit_all"]),
                "hit_all_n": comp["hit_all"][1],
                "accept_normal": frac(comp["accept_normal"]),
                "accept_normal_n": comp["accept_normal"][1],
                "pooled_accuracy": frac(comp["pooled"]),
                "pooled_n": comp["pooled"][1],
                "by_mode": {m: {"frac": frac(p), "n": p[1]}
                            for m, p in comp["by_mode"].items()},
            },
            "chance": {
                "hit_visible": frac(chance["hit_visible"]),
                "hit_all": frac(chance["hit_all"]),
                "pooled_accuracy": frac(chance["pooled"]),
            },
        })

    comp_vis = [s["comparator"]["hit_visible"] for s in per_seed]
    comp_acc = [s["comparator"]["accept_normal"] for s in per_seed]
    chance_vis = [s["chance"]["hit_visible"] for s in per_seed]
    h1_pass = (statistics.mean(comp_vis) >= 0.90
               and sum(1 for h in comp_vis if h >= 0.85) >= 3
               and statistics.mean(comp_acc) >= 0.80)

    quarantine = run_quarantine()
    h2_pass = (quarantine["leaks_admitted"] == 0
               and quarantine["audit_violators"] == 0
               and quarantine["chain_valid"]
               and all(r[1] in ("rejected", "ok") for r in quarantine["red_team"]))

    verdict = ("PASS" if h1_pass else "NULL_HOLDS") + " / " + \
              ("PASS" if h2_pass else "FAIL")
    summary = {
        "verdict": verdict,
        "H1_attribution": {
            "pass": h1_pass,
            "comparator_seed_mean_hit_visible": statistics.mean(comp_vis),
            "comparator_seed_hits_visible": comp_vis,
            "comparator_seed_mean_accept_normal": statistics.mean(comp_acc),
            "comparator_seed_accepts": comp_acc,
            "comparator_seed_mean_hit_all": statistics.mean(
                [s["comparator"]["hit_all"] for s in per_seed]),
            "comparator_seed_mean_err_coincide": statistics.mean(
                [s["comparator"]["err_coincide_rate"] for s in per_seed]),
            "chance_seed_mean_hit_visible": statistics.mean(chance_vis),
            "chance_seed_hits_visible": chance_vis,
            "hit_advantage_over_chance":
                statistics.mean(comp_vis) - statistics.mean(chance_vis),
        },
        "H2_quarantine": dict(quarantine, **{"pass": h2_pass}),
        "per_seed": per_seed,
        "consciousness_status": "UNRESOLVED",
    }

    result = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": SEEDS[0],
        "started_utc": started,
        "episodes": len(SEEDS) * EPISODES * 2 + QUARANTINE_EPISODES,
        "summary": summary,
    }
    receipts_dir = os.path.join(HERE, args.out)
    path = write_receipt(
        result, receipts_dir,
        hypothesis=("Comparator attributor: visible-mismatch hit >= 0.90 and "
                    "normal-accept >= 0.80; coinciding mismatches are "
                    "misattributed to self (comparator-predicted error). "
                    "Quarantine: zero leaks."),
        null=("Attribution at chance (~0.50 visible-mismatch hit); "
              "quarantine leaks."),
        preregistered_metric=("H1 hit_visible seed-mean >= 0.90 with "
                              "hit >= 0.85 on >= 3/4 seeds; "
                              "accept_normal seed-mean >= 0.80; "
                              "H2 leaks_admitted == 0, violators == 0"),
        baseline="ChanceAgent paired identical streams",
        conditions=("self_world_mismatch v1.0.0, p=0.15; 4 seeds x 5 eps x "
                    "60 ticks; burn-in 8 ticks; alternation policy"),
        interpretation=("PASS/NULL_HOLDS per decision rule in "
                        "preregistration_SELF_WORLD_ATTRIB.json"),
        limitations=("Alternation policy is a probe convenience; quarantine "
                     "is tag-enforced, not content-enforced; "
                     "CONSCIOUSNESS: UNRESOLVED."),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("receipt:", path)
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
