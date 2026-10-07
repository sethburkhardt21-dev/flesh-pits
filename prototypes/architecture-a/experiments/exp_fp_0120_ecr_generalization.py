"""EXP-FP-0120..0125 -- ECR GENERALIZATION BOUNDS battery (Track A).

Question: is ECR (Episodic Contrastive Return, the sparse-reward gain
redesign REPRODUCED 4/4 on canonical delayed_reward by EXP-FP-0080) a
general sparse-reward solution, or a delayed_reward-shaped one?

Protocol (per env): K5 P1 / EXP-FP-0080 identical in EVERY hyperparameter
except the task environment; the update rule is the INDEPENDENT ECR
reimplementation (ReturnConditionedEpisodicArbitrator in
attention_ecr_repro.py; attention.py / tick.py / module UNTOUCHED).

  Metric (bandit/corridor envs: changing_rule, compositional_rule,
  cue_delayed_reward, delayed_multistep): R = total(learned)/total(frozen)
  per seed. Gate: HOLDS iff R >= 1.30 on 4/4 fresh seeds.

  Metric (navigation envs: grid_world, pomaze): per-tick rewards are
  NEGATIVE (step penalties), so R = learned/frozen inverts meaning (a
  ratio of negatives). Primary metric = goal-reach counts per arm per
  seed. Gate: HOLDS iff learned_goal_episodes > frozen_goal_episodes on
  4/4 fresh seeds. Total return (less negative = better) reported as
  secondary. This gate adaptation is preregistered with justification.

Ablations (per env, descriptive, preregistered): the three ECR
sub-mechanisms on the FIRST 2 seeds of the env's seed set, reusing the
primary run's frozen totals (frozen is variant-independent: frozen=True
returns before any rate param matters, same as EXP-FP-0080):
  (a) no_boost     -- corridor_boost = 0.0
  (b) no_prebranch -- prebranch_demote = 0.0
  (c) no_punish    -- shape_punish = 0.0

Task-structural priors per env (documented, given not learned -- same
honesty as EXP-FP-0021/0080):
  grid_world / pomaze: every movement tick is a path-branching decision
      -> branch_channels = all movement channels (stay excluded).
      Sensitivity: the result is insensitive to this choice (zero or
      shaping-only update paths fire); verified by a one-seed re-run
      with branch_channels = {"north"} / {"north"} respectively.
  changing_rule / compositional_rule: every bandit tick is a branch
      decision -> branch_channels = {"a0", "a1"}.
  cue_delayed_reward / delayed_multistep: {"branch_a", "branch_b"}
      (delayed_reward identical).

Seeds: fresh per env (word-boundary grep: 0 hits across the flesh-pits
tree at preregistration time; see preregistration files).

Receipts: a separate harness-only writer (write_receipt_012x.py) reads
the staged per-env results and calls experiments/harness.py
write_receipt (hash-chained, fail-closed). The split keeps the
architecture-a and experiments/envs import domains from colliding on the
module name `envs`.
"""
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)  # prototypes/architecture-a
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention_ecr_repro import ReturnConditionedEpisodicArbitrator
from grid_world import GridWorld
from pomaze import POMaze
from changing_rule import ChangingRule
from compositional_rule import CompositionalRule
from cue_delayed_reward import CueDelayedReward
from delayed_multistep import DelayedMultistep
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)
from env_interface import config_hash, make_episode_id

THETA = 0.45
MARGIN = 1.30
GAIN_LR = 0.15
NEED = 4

# env_name: (experiment_id, env_cls, seeds, n_episodes, branch_channels)
ENVS = {
    "grid_world": (
        "EXP-FP-0120", GridWorld, (87101, 87102, 87103, 87104), 8,
        {"north", "south", "east", "west"}, "nav"),
    "pomaze": (
        "EXP-FP-0121", POMaze, (87201, 87202, 87203, 87204), 6,
        {"north", "south", "east", "west"}, "nav"),
    "changing_rule": (
        "EXP-FP-0122", ChangingRule, (87301, 87302, 87303, 87304), 12,
        {"a0", "a1"}, "bandit"),
    "compositional_rule": (
        "EXP-FP-0123", CompositionalRule, (87401, 87402, 87403, 87404), 12,
        {"a0", "a1"}, "bandit"),
    "cue_delayed_reward": (
        "EXP-FP-0124", CueDelayedReward, (87501, 87502, 87503, 87504), 8,
        {"branch_a", "branch_b"}, "corridor"),
    "delayed_multistep": (
        "EXP-FP-0125", DelayedMultistep, (87601, 87602, 87603, 87604), 8,
        {"branch_a", "branch_b"}, "corridor"),
}

BASE_ARBITRATOR_KWARGS = {
    "shape_punish": 0.02,
    "prebranch_demote": 0.05,
    "corridor_boost": 0.01,
}


class GoalCountingAdapter(CanonicalAdapter):
    """CanonicalAdapter + goal-episode counting via env info.

    k5_multitask_generalization.py is a canonical file and stays
    byte-identical (G0); the counting lives in this additive subclass.
    info["goal_reached"] is read for COUNTING ONLY (analysis), never for
    decisions -- the agent path is unchanged.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.goal_episodes = 0

    def step(self, action_label):
        a = self.channel_to_action[action_label]
        obs, reward, done, info = self.env.step(a)
        self.obs = obs
        if done:
            if info.get("goal_reached"):
                self.goal_episodes += 1
            self.episodes_done += 1
            self.ep += 1
            self.obs = self.env.reset(
                self.episode_seeds[self.ep % len(self.episode_seeds)])
        return float(reward)


def probe(env_name, seed, frozen, arb_cls=None, arb_extra=None):
    """EXP-FP-0080 probe protocol, generalized over the env table."""
    exp_id, env_cls, seeds, n_episodes, branch_channels, kind = \
        ENVS[env_name]
    env_obj = env_cls()
    channels = list(env_obj.action_space()["labels"])
    max_steps = env_obj.MAX_STEPS if hasattr(env_obj, "MAX_STEPS") \
        else env_obj.STEPS
    kwargs = dict(BASE_ARBITRATOR_KWARGS)
    kwargs["branch_channels"] = set(branch_channels)
    kwargs["max_steps"] = max_steps
    kwargs.update(arb_extra or {})
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=arb_cls
                       or ReturnConditionedEpisodicArbitrator,
                       arbitrator_kwargs=kwargs)
    assert isinstance(wk.arbitrator, ReturnConditionedEpisodicArbitrator)
    env = GoalCountingAdapter(env_obj, seed,
                              {c: i for i, c in enumerate(channels)},
                              n_episodes=n_episodes)
    n_success = 0
    for _ in range(n_episodes * max_steps):
        tr = wk.step(env.observe(), env)
        if tr.get("reward", 0.0) >= 1.0:
            n_success += 1
    return (sum(wk.rewards), dict(wk.arbitrator.gains),
            len(wk.arbitrator.gain_history), n_success,
            env.goal_episodes, env.episodes_done,
            channels, max_steps, n_episodes)


def seed_win(env_name, learned_total, frozen_total,
             learned_goals, frozen_goals):
    """Per-seed WIN per the preregistered env-class gate.

    nav: primary metric is goal-reaching episodes (per-tick rewards are
    negative, so the R ratio inverts meaning). Otherwise R >= 1.30.
    """
    kind = ENVS[env_name][5]
    if kind == "nav":
        return learned_goals > frozen_goals
    return (learned_total / frozen_total) >= MARGIN \
        if frozen_total > 0 else False


def run_env(env_name):
    exp_id, env_cls, seeds, n_episodes, branch_channels, kind = \
        ENVS[env_name]
    env_probe = env_cls()
    max_steps = env_probe.MAX_STEPS if hasattr(env_probe, "MAX_STEPS") \
        else env_probe.STEPS
    print(f"{exp_id} [{env_name} v{env_probe.VERSION}]: ECR learned vs "
          f"frozen, {len(seeds)} seeds x {n_episodes} eps x <={max_steps} "
          f"steps; kind={kind}")

    episodes = []
    per_seed = []
    wins = 0
    frozen_totals = {}
    frozen_goals_map = {}
    for seed in seeds:
        tl, gains_l, n_upd, n_suc_l, n_goals_l, n_eps_l, channels, ms, ne = \
            probe(env_name, seed, False)
        tf, _, _, n_suc_f, n_goals_f, n_eps_f, _, _, _ = \
            probe(env_name, seed, True)
        frozen_totals[seed] = tf
        frozen_goals_map[seed] = n_goals_f
        r = (tl / tf) if tf > 0 else (float("inf") if tf == 0 else None)
        win = seed_win(env_name, tl, tf, n_goals_l, n_goals_f)
        wins += int(win)
        per_seed.append({
            "seed": seed,
            "learned_total": tl, "frozen_total": tf,
            "ratio": r, "win": bool(win),
            "learned_successes": n_suc_l, "frozen_successes": n_suc_f,
            "learned_goal_episodes": n_goals_l,
            "frozen_goal_episodes": n_goals_f,
            "learned_env_episodes": n_eps_l,
            "frozen_env_episodes": n_eps_f,
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
            "n_gain_updates": n_upd,
        })
        for (arm, total, n_suc, n_goals, n_eps) in (
                ("learned", tl, n_suc_l, n_goals_l, n_eps_l),
                ("frozen", tf, n_suc_f, n_goals_f, n_eps_f)):
            episodes.append({
                "episode_id": make_episode_id(
                    env_probe.NAME, env_probe.VERSION, seed,
                    0 if arm == "learned" else 1),
                "seed": seed, "arm": arm, "n_env_episodes": n_eps,
                "max_steps": ms,
                "return": total, "successes": n_suc,
                "goal_episodes": n_goals,
                "truncated": False, "done": True,
            })
        print(f"  seed={seed}: learned={tl:.2f} (succ={n_suc_l}, "
              f"goals={n_goals_l}/{n_eps_l}) "
              f"frozen={tf:.2f} (succ={n_suc_f}, goals={n_goals_f}/{n_eps_f}) "
              f"R={'%.2f' % r if r is not None else 'n/a'} "
              f"{'WIN' if win else 'LOSS'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    # --- ablations: three sub-mechanisms, first 2 seeds, descriptive ---
    abl_seeds = seeds[:2]
    variants = {
        "no_boost": {"corridor_boost": 0.0},
        "no_prebranch": {"prebranch_demote": 0.0},
        "no_punish": {"shape_punish": 0.0},
    }
    ablations = {}
    for vname, extra in variants.items():
        arm = {"seeds": {}, "wins": 0}
        for seed in abl_seeds:
            tl, _, _, n_suc_l, n_goals_l, _, _, _, _ = probe(
                env_name, seed, False, arb_extra=extra)
            tf = frozen_totals[seed]
            r = (tl / tf) if tf > 0 else (float("inf") if tf == 0 else None)
            win = seed_win(env_name, tl, tf, n_goals_l,
                           frozen_goals_map[seed])
            arm["wins"] += int(win)
            arm["seeds"][str(seed)] = {
                "learned_total": tl, "frozen_total": tf,
                "ratio": r, "win": bool(win),
                "learned_successes": n_suc_l,
                "learned_goal_episodes": n_goals_l,
                "frozen_goal_episodes": frozen_goals_map[seed],
            }
            print(f"    {vname:12s} seed={seed}: learned={tl:.2f} "
                  f"(succ={n_suc_l}) R="
                  f"{'%.2f' % r if r is not None else 'n/a'} "
                  f"{'WIN' if win else 'LOSS'}")
        arm["note"] = "descriptive, not gated; frozen totals reused"
        ablations[vname] = arm

    passed = (wins == NEED)
    verdict = "HOLDS" if passed else "BREAKS"
    print(f"  -> {verdict} ({wins}/{NEED})")

    returns = [e["return"] for e in episodes]
    config = {
        "env": f"{env_probe.NAME} v{env_probe.VERSION}",
        "episodes_per_arm": n_episodes,
        "max_steps": max_steps,
        "channels": list(env_probe.action_space()["labels"]),
        "branch_channels": sorted(branch_channels),
        "specialists": "neutral (K5 P1 identical)",
        "capacity": 4, "gain_lr": GAIN_LR, "theta": THETA,
        "gain_cap": 2.0, "floor": 0.01,
        "arbitrator": "ReturnConditionedEpisodicArbitrator "
                      "(EXP-FP-0080 independent reimplementation, module "
                      "UNTOUCHED)",
        "arbitrator_params": {"branch_channels": sorted(branch_channels),
                              "max_steps": max_steps, **{
                                  k: v for k, v in
                                  BASE_ARBITRATOR_KWARGS.items()}},
        "frozen": "same class, frozen=True, gains pinned 1.0",
        "gate": ("learned_goal_episodes > frozen_goal_episodes on 4/4 "
                 "(nav: negative per-tick rewards invert the R ratio)"
                 if kind == "nav" else "R >= 1.30 on 4/4"),
        "seeds": list(seeds),
        "replicates": "EXP-FP-0080 (cross-env generalization)",
    }
    result = {
        "experiment_id": exp_id,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": seeds[0],
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "episodes": episodes,
        "summary": {
            "n_episodes": len(episodes),
            "mean_return": statistics.fmean(returns),
            "stdev_return": statistics.pstdev(returns)
            if len(returns) > 1 else 0.0,
            "min_return": min(returns), "max_return": max(returns),
            "per_seed": per_seed,
            "wins": wins, "need": NEED, "margin": MARGIN,
            "passed": bool(passed),
            "verdict": verdict,
            "ablations": ablations,
        },
    }
    results_path = os.path.join(
        FLESH, "var", f"ecr-gen-{env_name}-results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("staged results:", results_path)

    # G2 determinism spot-check: recompute the first seed's learned arm.
    tl2, _, _, _, _, _, _, _, _ = probe(env_name, seeds[0], False)
    tl1 = per_seed[0]["learned_total"]
    assert abs(tl2 - tl1) < 1e-12, f"G2 FAIL [{env_name}]: {tl2} != {tl1}"
    print(f"G2: determinism spot-check PASS [{env_name}] (seed "
          f"{seeds[0]} learned={tl1:.6f} exact)")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None,
                    help="run one env (default: all six)")
    args = ap.parse_args(argv)
    targets = [args.env] if args.env else list(ENVS)
    for env_name in targets:
        if env_name not in ENVS:
            raise SystemExit(f"unknown env {env_name!r}")
        run_env(env_name)


if __name__ == "__main__":
    main()
