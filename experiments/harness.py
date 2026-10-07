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
CHAIN CONVENTION (2026-10-07, bug fix):
- write_receipt and verify_chain both operate on the *chained subsequence*:
  only receipts that carry a receipt_hash link into the chain, in mtime order.
  Pre-chain (hashless, legacy) receipts are ignored by BOTH when computing
  linkage, so write and verify always agree.
- Rationale: verify_chain is the checker, so write must conform to it; a
  prev_receipt_hash of None is only legitimate when no chained receipt
  precedes the new one. Chaining to a hashless file anchors to nothing
  (there is no receipt_hash to bind to), so excluding pre-chain files loses
  no integrity and removes the old write/verify disagreement (the EXP-AB-K3C
  lane confirmed the old mismatch live).
- Existing receipts are never rewritten; only future writes use this rule.

OVERWRITE RULE (2026-10-07, EXP-FP-0050 collision):
- write_receipt FAILS CLOSED on an existing path (raises ReceiptExistsError)
  unless supersede="<reason>" is passed; the superseded receipt records
  superseded_previous_hash + supersede_reason. Claim IDs in
  experiments/ID_REGISTRY (id_registry.py, atomic claims under the
  file-lock skill) before writing; never mint blind.
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


class ReceiptExistsError(FileExistsError):
    """Raised when write_receipt refuses to overwrite an existing receipt.

    Subclasses FileExistsError so existing callers catching FileExistsError
    also catch this."""


def _chained_receipt_files(receipts_dir, exclude_path=None):
    """mtime-sorted .json files that carry a receipt_hash.

    The chain lives on this subsequence (see CHAIN CONVENTION at top).
    Files that are unreadable, invalid JSON, or hashless are not chained.
    """
    files = sorted(
        (q for q in os.listdir(receipts_dir) if q.endswith(".json")),
        key=lambda q: os.path.getmtime(os.path.join(receipts_dir, q)))
    chained = []
    for fn in files:
        p = os.path.join(receipts_dir, fn)
        if exclude_path is not None and p == exclude_path:
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("receipt_hash") is None:
            continue
        chained.append(fn)
    return chained


def write_receipt(result, receipts_dir, hypothesis="", null="",
                  preregistered_metric="", baseline="", conditions="",
                  interpretation="", limitations="", supersede=None):
    """Write the §38/§46 receipt JSON. Returns the file path.

    Receipts are hash-chained: each receipt carries prev_receipt_hash (sha256
    of the most recently written *chained* receipt in receipts_dir, None when
    no chained receipt precedes it) and its own receipt_hash. verify_chain()
    checks the chain. Pre-chain (hashless) receipts are skipped when choosing
    prev — see CHAIN CONVENTION at module top.

    FAIL-CLOSED OVERWRITE RULE (2026-10-07, EXP-FP-0050 collision):
    if <receipts_dir>/<experiment_id>.json already exists, the write is
    REFUSED with FileExistsError — never silently overwritten. This is the
    systemic fix for the 2026-10-07 incident where two concurrent lanes
    minted EXP-FP-0050 and the second lane's open(path, "w") destroyed the
    first lane's receipt (hash c55fdebe..., unrecoverable). Claim the ID in
    experiments/ID_REGISTRY before writing; never mint blind.

    To deliberately replace an existing receipt (e.g. re-run after a crash
    truncated the first write), pass supersede="<non-empty reason>": the
    write proceeds, and the receipt records superseded_previous_hash (the
    old receipt's receipt_hash, None if it was hashless) and
    supersede_reason. An empty or missing reason still refuses.
    """
    import hashlib
    os.makedirs(receipts_dir, exist_ok=True)
    path = os.path.join(receipts_dir, f"{result['experiment_id']}.json")
    path_exists = os.path.exists(path)
    reason = supersede if isinstance(supersede, str) and supersede else None
    if path_exists and reason is None:
        raise ReceiptExistsError(
            f"refusing to overwrite existing receipt {path}: the experiment "
            "ID is already taken (possible concurrent-lane collision — "
            "claim the ID family in experiments/ID_REGISTRY first and never "
            "mint blind). Pass supersede='<reason>' to deliberately "
            "supersede the existing receipt.")
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
    if path_exists:
        # Explicit supersede: bind the replacement to what it replaces.
        try:
            with open(path) as f:
                old = json.load(f)
            receipt["superseded_previous_hash"] = old.get("receipt_hash")
        except (json.JSONDecodeError, OSError):
            receipt["superseded_previous_hash"] = None
        receipt["supersede_reason"] = reason
    prev = None
    chained = _chained_receipt_files(receipts_dir, exclude_path=path)
    if chained:
        # chain to the previous chained receipt's content hash
        with open(os.path.join(receipts_dir, chained[-1])) as f:
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
    prev_receipt_hash matches the previous *chained* receipt's stored
    receipt_hash (None for the first chained receipt). Pre-chain receipts
    (no receipt_hash) are reported, not failed, and are excluded from the
    linkage sequence — matching write_receipt (CHAIN CONVENTION, module top).
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
