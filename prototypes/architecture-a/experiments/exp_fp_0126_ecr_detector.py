"""EXP-FP-0126 -- adaptive terminal detector for ECR (Track A, Phase 8).

Follows EXP-FP-0120..0125's battery protocol (driver derived from
exp_fp_0120_ecr_generalization.py; that driver is untouched): K5 P1 /
EXP-FP-0080 identical in every hyperparameter except the arbitrator
class, which is the additive AdaptiveTerminalECR subclass with a
detector= kwarg. attention_ecr_repro.py / attention.py / tick.py /
envs are UNTOUCHED.

Two envs:
  grid_world    (nav, 8 eps x <=100 steps, branch = all movement
                 channels, GoalCountingAdapter): the 0120 failure site.
  delayed_reward (corridor, 8 eps x <=15 steps, branch =
                 {branch_a, branch_b}, CanonicalAdapter): the home env
                 -- regression gate.

Main arms (gated): D3 (relmax) learned vs frozen, 4 fresh seeds per env.
Ablations (descriptive): on the FIRST 2 seeds per env, the other
detectors' learned arms reuse the main run's frozen totals (frozen is
detector-independent: frozen=True returns before classification).
  grid_world:    D0 (hardcoded), D1 (floor), D2 (baseline)
  delayed_reward: D0, D1, D2  (D0 = the 0080 rule verbatim, reference)
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
from attention_ecr_detector import AdaptiveTerminalECR
from grid_world import GridWorld
from delayed_reward import DelayedReward
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)
from env_interface import config_hash, make_episode_id

THETA = 0.45
MARGIN = 1.30
GAIN_LR = 0.15
NEED = 4

DETECTORS = ["hardcoded", "floor", "baseline", "relmax"]
MAIN = "relmax"  # D3: the H1 detector

# env_name: (env_cls, seeds, n_episodes, branch_channels, kind, adapter)
ENVS = {
    "grid_world": (
        GridWorld, (88101, 88102, 88103, 88104), 8,
        {"north", "south", "east", "west"}, "nav"),
    "delayed_reward": (
        DelayedReward, (88201, 88202, 88203, 88204), 8,
        {"branch_a", "branch_b"}, "corridor"),
}

BASE_ARBITRATOR_KWARGS = {
    "shape_punish": 0.02,
    "prebranch_demote": 0.05,
    "corridor_boost": 0.01,
}


class GoalCountingAdapter(CanonicalAdapter):
    """CanonicalAdapter + goal-episode counting via env info.

    k5_multitask_generalization.py stays byte-identical (G0); the
    counting lives here (additive subclass, same as the 0120 battery).
    info["goal_reached"] is read for COUNTING ONLY, never for decisions.
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


def probe(env_name, seed, frozen, detector):
    """0120 probe protocol with the detector kwarg."""
    env_cls, seeds, n_episodes, branch_channels, kind = ENVS[env_name]
    env_obj = env_cls()
    channels = list(env_obj.action_space()["labels"])
    max_steps = env_obj.MAX_STEPS if hasattr(env_obj, "MAX_STEPS") \
        else env_obj.STEPS
    kwargs = dict(BASE_ARBITRATOR_KWARGS)
    kwargs["branch_channels"] = set(branch_channels)
    kwargs["max_steps"] = max_steps
    kwargs["detector"] = detector
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=AdaptiveTerminalECR,
                       arbitrator_kwargs=kwargs)
    assert isinstance(wk.arbitrator, AdaptiveTerminalECR)
    adapter_cls = GoalCountingAdapter if kind == "nav" else CanonicalAdapter
    env = adapter_cls(env_obj, seed,
                      {c: i for i, c in enumerate(channels)},
                      n_episodes=n_episodes)
    for _ in range(n_episodes * max_steps):
        wk.step(env.observe(), env)
    n_goals = getattr(env, "goal_episodes", None)
    return (sum(wk.rewards), dict(wk.arbitrator.gains),
            len(wk.arbitrator.gain_history),
            n_goals, env.episodes_done,
            channels, max_steps, n_episodes)


def seed_win(env_name, learned_total, frozen_total,
             learned_goals, frozen_goals):
    """Per-seed WIN per the preregistered env-class gate."""
    kind = ENVS[env_name][4]
    if kind == "nav":
        return learned_goals > frozen_goals
    return (learned_total / frozen_total) >= MARGIN \
        if frozen_total > 0 else False


def run_env(env_name):
    env_cls, seeds, n_episodes, branch_channels, kind = ENVS[env_name]
    env_probe = env_cls()
    max_steps = env_probe.MAX_STEPS if hasattr(env_probe, "MAX_STEPS") \
        else env_probe.STEPS
    print(f"EXP-FP-0126 [{env_name} v{env_probe.VERSION}]: D3(relmax) "
          f"learned vs frozen + D0/D1/D2 ablations, {len(seeds)} seeds x "
          f"{n_episodes} eps x <={max_steps} steps; kind={kind}")

    per_seed = []
    wins = 0
    frozen_totals = {}
    frozen_goals_map = {}
    frozen_gains_map = {}
    for seed in seeds:
        tl, gains_l, n_upd, n_goals_l, n_eps_l, channels, ms, ne = \
            probe(env_name, seed, False, MAIN)
        tf, gains_f, _, n_goals_f, n_eps_f, _, _, _ = \
            probe(env_name, seed, True, MAIN)
        frozen_totals[seed] = tf
        frozen_goals_map[seed] = n_goals_f
        frozen_gains_map[seed] = {k: round(v, 3)
                                  for k, v in gains_f.items()}
        win = seed_win(env_name, tl, tf, n_goals_l, n_goals_f)
        wins += int(win)
        r = (tl / tf) if tf > 0 else (float("inf") if tf == 0 else None)
        per_seed.append({
            "seed": seed,
            "learned_total": tl, "frozen_total": tf,
            "ratio": r, "win": bool(win),
            "learned_goal_episodes": n_goals_l,
            "frozen_goal_episodes": n_goals_f,
            "learned_env_episodes": n_eps_l,
            "frozen_env_episodes": n_eps_f,
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
            "n_gain_updates": n_upd,
        })
        print(f"  seed={seed}: D3 learned={tl:.2f} (goals={n_goals_l}/"
              f"{n_eps_l}) frozen={tf:.2f} (goals={n_goals_f}/{n_eps_f}) "
              f"R={'%.2f' % r if r not in (None,) and r != float('inf') else 'n/a'} "
              f"{'WIN' if win else 'LOSS'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    # --- ablations: D0/D1/D2 learned, first 2 seeds, frozen reused ---
    ablations = {}
    for det in [d for d in DETECTORS if d != MAIN]:
        arm = {"seeds": {}, "wins": 0}
        for seed in seeds[:2]:
            tl, gains_l, n_upd, n_goals_l, _, _, _, _ = \
                probe(env_name, seed, False, det)
            tf = frozen_totals[seed]
            win = seed_win(env_name, tl, tf, n_goals_l,
                           frozen_goals_map[seed])
            arm["wins"] += int(win)
            r = (tl / tf) if tf > 0 else (float("inf") if tf == 0 else None)
            arm["seeds"][str(seed)] = {
                "learned_total": tl, "frozen_total": tf,
                "ratio": r, "win": bool(win),
                "learned_goal_episodes": n_goals_l,
                "frozen_goal_episodes": frozen_goals_map[seed],
                "learned_gains": {k: round(v, 3)
                                  for k, v in gains_l.items()},
                "n_gain_updates": n_upd,
            }
            print(f"    {det:9s} seed={seed}: learned={tl:.2f} "
                  f"(goals={n_goals_l}) "
                  f"{'WIN' if win else 'LOSS'} gains="
                  f"{ {k: round(v, 2) for k, v in gains_l.items()} }")
        arm["note"] = ("descriptive, not gated; frozen totals reused "
                       "from the D3 main run")
        ablations[det] = arm

    passed = (wins == NEED)
    verdict = "HOLDS" if passed else "BREAKS"
    print(f"  -> {verdict} ({wins}/{NEED})")

    config = {
        "env": f"{env_probe.NAME} v{env_probe.VERSION}",
        "episodes_per_arm": n_episodes,
        "max_steps": max_steps,
        "channels": list(env_probe.action_space()["labels"]),
        "branch_channels": sorted(branch_channels),
        "specialists": "neutral (K5 P1 identical)",
        "capacity": 4, "gain_lr": GAIN_LR, "theta": THETA,
        "gain_cap": 2.0, "floor": 0.01,
        "arbitrator": "AdaptiveTerminalECR (EXP-FP-0126 additive subclass "
                      "of ReturnConditionedEpisodicArbitrator; canonical "
                      "module UNTOUCHED)",
        "main_detector": MAIN,
        "detector_params": {"FLOOR": 0.1, "REL": 0.5, "K": 2.0,
                            "ALPHA": 0.1},
        "arbitrator_params": {"branch_channels": sorted(branch_channels),
                              "max_steps": max_steps, **{
                                  k: v for k, v in
                                  BASE_ARBITRATOR_KWARGS.items()}},
        "frozen": "same class, frozen=True, gains pinned 1.0",
        "gate": ("learned_goal_episodes > frozen_goal_episodes on 4/4 "
                 "(nav: negative per-tick rewards invert the R ratio)"
                 if kind == "nav" else "R >= 1.30 on 4/4"),
        "seeds": list(seeds),
        "ablations": sorted(d for d in DETECTORS if d != MAIN),
    }
    result = {
        "experiment_id": "EXP-FP-0126",
        "env": env_name,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": seeds[0],
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "summary": {
            "per_seed": per_seed,
            "wins": wins, "need": NEED, "margin": MARGIN,
            "passed": bool(passed),
            "verdict": verdict,
            "ablations": ablations,
        },
    }
    results_path = os.path.join(
        FLESH, "var", f"ecr-detector-{env_name}-results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("staged results:", results_path)

    # G2 determinism spot-check: recompute the first seed's D3 learned arm.
    tl2, _, _, _, _, _, _, _ = probe(env_name, seeds[0], False, MAIN)
    tl1 = per_seed[0]["learned_total"]
    assert abs(tl2 - tl1) < 1e-12, f"G2 FAIL [{env_name}]: {tl2} != {tl1}"
    print(f"G2: determinism spot-check PASS [{env_name}] (seed "
          f"{seeds[0]} D3 learned={tl1:.6f} exact)")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None,
                    help="run one env (default: both)")
    args = ap.parse_args(argv)
    targets = [args.env] if args.env else list(ENVS)
    for env_name in targets:
        if env_name not in ENVS:
            raise SystemExit(f"unknown env {env_name!r}")
        run_env(env_name)


if __name__ == "__main__":
    main()
