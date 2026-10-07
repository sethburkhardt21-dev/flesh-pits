"""Experiment harness — run episodes, run experiments, write receipts.

Usage:
    python3 harness.py --env grid_world --agent arch_d --episodes 30 \\
        --primary-seed 7 --experiment-id EXP-FP-0001

Conventions:
- One env instance is reused across all episodes of a run, so
  episode-indexed dynamics (e.g. changing_rule phase flips) advance.
- One agent instance is reused across all episodes: cross-episode learning
  accumulates in the agent object (per ENV_INTERFACE.md).
- Per-episode seeds derive deterministically from (primary_seed, run_index)
  via env_interface.derive_seed; agent seeds use the "agent" stream.
- Stdlib only.
"""

import argparse
import datetime
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "envs"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototypes"))

from env_interface import (  # noqa: E402
    CONTRACT_VERSION, config_hash, derive_seed, make_episode_id,
)
from envs import ALL_ENVS  # noqa: E402
from baselines import ALL_BASELINES  # noqa: E402
from arch_d import ArchD  # noqa: E402

ALL_AGENTS = ALL_BASELINES + [ArchD]
ENVS_BY_NAME = {c.NAME: c for c in ALL_ENVS}
AGENTS_BY_NAME = {c.NAME: c for c in ALL_AGENTS}


def run_episode(env, agent, episode_seed, run_index, max_steps=None,
                validate=False):
    """Run one episode. Returns a summary dict (no full trajectory by default)."""
    episode_id = make_episode_id(env.NAME, env.VERSION, episode_seed, run_index)
    obs = env.reset(episode_seed)
    if validate:
        env.validate_obs(obs, env.observation_space())
    init_hash = env.state_hash()
    agent.reset(derive_seed(episode_seed, run_index, "agent"),
                env.action_space())
    total, steps = 0.0, 0
    done, info = False, {}
    while not done:
        if max_steps is not None and steps >= max_steps:
            info = {"truncated": True, "harness_cap": True}
            break
        action = agent.act(obs)
        env.validate_action(action)
        obs2, reward, done, info = env.step(action)
        if validate:
            env.validate_obs(obs2, env.observation_space())
        agent.update(obs, action, reward, done, info)
        total += reward
        steps += 1
        obs = obs2
    return {
        "episode_id": episode_id,
        "seed": episode_seed,
        "init_hash": init_hash,
        "final_hash": env.state_hash(),
        "return": total,
        "steps": steps,
        "done": done,
        "truncated": bool(info.get("truncated", False)),
    }


def run_experiment(env_name, agent_name, n_episodes, primary_seed,
                   experiment_id, max_steps=None, validate=False):
    """Run n_episodes; agent learns across episodes; env phases advance."""
    env_cls = ENVS_BY_NAME[env_name]
    agent_cls = AGENTS_BY_NAME[agent_name]
    env, agent = env_cls(), agent_cls()  # reused across episodes, by design
    episodes = []
    for run_index in range(n_episodes):
        episode_seed = derive_seed(primary_seed, run_index, "env")
        episodes.append(run_episode(env, agent, episode_seed, run_index,
                                    max_steps=max_steps, validate=validate))
    returns = [e["return"] for e in episodes]
    config = {
        "contract_version": CONTRACT_VERSION,
        "env": env_name, "env_version": env.VERSION,
        "agent": agent_name,
        "n_episodes": n_episodes,
        "primary_seed": primary_seed,
        "max_steps": max_steps,
    }
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "experiment_id": experiment_id,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": primary_seed,
        "started_utc": now,
        "episodes": episodes,
        "summary": {
            "n_episodes": n_episodes,
            "mean_return": statistics.fmean(returns),
            "stdev_return": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
            "min_return": min(returns),
            "max_return": max(returns),
            "mean_steps": statistics.fmean(e["steps"] for e in episodes),
        },
    }


def write_receipt(result, receipts_dir, hypothesis="", null="",
                  preregistered_metric="", baseline="", conditions="",
                  interpretation="", limitations=""):
    """Write the §38/§46 receipt JSON. Returns the file path.

    Receipts are hash-chained: each receipt carries prev_receipt_hash (sha256
    of the most recently written *other* receipt in receipts_dir, None for the
    first) and its own receipt_hash. verify_chain() checks the chain.
    """
    import hashlib
    os.makedirs(receipts_dir, exist_ok=True)
    receipt = {
        "experiment_id": result["experiment_id"],
        "hypothesis": hypothesis,
        "null_hypothesis": null,
        "preregistered_metric": preregistered_metric,
        "baseline": baseline,
        "conditions": conditions,
        "config": result["config"],
        "config_hash": result["config_hash"],
        "primary_seed": result["primary_seed"],
        "started_utc": result["started_utc"],
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "episodes": result["episodes"],
        "summary": result["summary"],
        "interpretation": interpretation,
        "limitations": limitations,
    }
    path = os.path.join(receipts_dir, f"{result['experiment_id']}.json")
    prev = None
    candidates = sorted(
        (q for q in os.listdir(receipts_dir) if q.endswith(".json")),
        key=lambda q: os.path.getmtime(os.path.join(receipts_dir, q)))
    candidates = [c for c in candidates
                  if os.path.join(receipts_dir, c) != path]
    if candidates:
        # chain to the previous receipt's content hash (None if pre-chain)
        with open(os.path.join(receipts_dir, candidates[-1])) as f:
            try:
                prev = json.load(f).get("receipt_hash")
            except (json.JSONDecodeError, OSError):
                prev = None
    receipt["prev_receipt_hash"] = prev
    body = json.dumps(receipt, indent=2, sort_keys=True)
    receipt["receipt_hash"] = hashlib.sha256(body.encode()).hexdigest()
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


def verify_chain(receipts_dir):
    """Verify the hash chain over *.json receipts sorted by mtime.

    Returns (ok, problems). A receipt verifies iff its stored receipt_hash
    matches the hash of its body minus receipt_hash, and its
    prev_receipt_hash matches the previous receipt's stored receipt_hash
    (None for the first). Pre-chain receipts (no receipt_hash) are reported,
    not failed.
    """
    import hashlib
    files = sorted(
        (q for q in os.listdir(receipts_dir) if q.endswith(".json")),
        key=lambda q: os.path.getmtime(os.path.join(receipts_dir, q)))
    problems = []
    prev_hash = None
    for fn in files:
        p = os.path.join(receipts_dir, fn)
        with open(p) as f:
            rec = json.load(f)
        stored = rec.get("receipt_hash")
        if stored is None:
            problems.append(f"{fn}: no receipt_hash (pre-chain receipt)")
            continue
        body = {k: v for k, v in rec.items() if k != "receipt_hash"}
        calc = hashlib.sha256(
            json.dumps(body, indent=2, sort_keys=True).encode()).hexdigest()
        if calc != stored:
            problems.append(f"{fn}: receipt_hash mismatch")
        if rec.get("prev_receipt_hash") != prev_hash:
            problems.append(f"{fn}: prev_receipt_hash mismatch")
        prev_hash = stored
    return (len(problems) == 0, problems)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Flesh Pits experiment harness")
    ap.add_argument("--env", required=True, choices=sorted(ENVS_BY_NAME))
    ap.add_argument("--agent", required=True, choices=sorted(AGENTS_BY_NAME))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--primary-seed", type=int, default=1)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--receipts", default=None,
                    help="receipts dir (default: <tree>/receipts)")
    ap.add_argument("--validate", action="store_true",
                    help="validate every obs against its space (slower)")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--null", default="")
    ap.add_argument("--metric", default="mean_return")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--conditions", default="")
    args = ap.parse_args(argv)

    receipts = args.receipts or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "receipts")
    result = run_experiment(args.env, args.agent, args.episodes,
                            args.primary_seed, args.experiment_id,
                            max_steps=args.max_steps, validate=args.validate)
    path = write_receipt(result, receipts, hypothesis=args.hypothesis,
                         null=args.null, preregistered_metric=args.metric,
                         baseline=args.baseline, conditions=args.conditions)
    s = result["summary"]
    print(f"{args.experiment_id}: {args.agent} x {args.env} "
          f"n={s['n_episodes']} mean_return={s['mean_return']:.4f} "
          f"stdev={s['stdev_return']:.4f} config_hash={result['config_hash'][:12]}")
    print(f"receipt: {path}")
    return path


if __name__ == "__main__":
    main()
