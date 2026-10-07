"""EXP-FP-0127 -- progress-tie-break guard (EXP-FP-0126 follow-up).

Question: is the arbitrary progress pin (canonical tie-break on zero
shaping receipts) THE harmful half of the ECR lock-in on navigation,
or is the lock-in concept itself mismatched?

Arms: D3 (relmax) + progress_guard=True, learned vs frozen, grid_world,
4 fresh seeds {88301..88304}, 8 eps x <=100 steps, 0126 probe protocol
(GoalCountingAdapter). Bit-identity check: delayed_reward seeds
88201/88202 (reused -- code-equivalence, not a new measurement):
D3+guard learned totals must equal the EXP-FP-0126 D3 totals exactly
(the guard can never trigger where shaping exists).
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))
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
from env_interface import config_hash

THETA = 0.45
GAIN_LR = 0.15
NEED = 4
SEEDS = (88301, 88302, 88303, 88304)
BITID_SEEDS = (88201, 88202)

KWARGS = {
    "shape_punish": 0.02,
    "prebranch_demote": 0.05,
    "corridor_boost": 0.01,
    "detector": "relmax",
    "progress_guard": True,
}


class GoalCountingAdapter(CanonicalAdapter):
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


def probe(env_cls, seed, frozen, n_episodes, branch_channels, nav):
    env_obj = env_cls()
    channels = list(env_obj.action_space()["labels"])
    max_steps = env_obj.MAX_STEPS
    kwargs = dict(KWARGS)
    kwargs["branch_channels"] = set(branch_channels)
    kwargs["max_steps"] = max_steps
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=AdaptiveTerminalECR,
                       arbitrator_kwargs=kwargs)
    assert isinstance(wk.arbitrator, AdaptiveTerminalECR)
    adapter_cls = GoalCountingAdapter if nav else CanonicalAdapter
    env = adapter_cls(env_obj, seed,
                      {c: i for i, c in enumerate(channels)},
                      n_episodes=n_episodes)
    for _ in range(n_episodes * max_steps):
        wk.step(env.observe(), env)
    return (sum(wk.rewards), dict(wk.arbitrator.gains),
            getattr(env, "goal_episodes", None), env.episodes_done)


def main():
    print("EXP-FP-0127 [grid_world v1.0.0]: D3+guard learned vs frozen, "
          f"{len(SEEDS)} fresh seeds")
    per_seed = []
    no_harm = 0
    for seed in SEEDS:
        tl, gains_l, n_goals_l, n_eps_l = probe(
            GridWorld, seed, False, 8,
            {"north", "south", "east", "west"}, True)
        tf, _, n_goals_f, n_eps_f = probe(
            GridWorld, seed, True, 8,
            {"north", "south", "east", "west"}, True)
        ok = n_goals_l >= n_goals_f
        no_harm += int(ok)
        per_seed.append({
            "seed": seed,
            "learned_total": tl, "frozen_total": tf,
            "learned_goal_episodes": n_goals_l,
            "frozen_goal_episodes": n_goals_f,
            "learned_env_episodes": n_eps_l,
            "frozen_env_episodes": n_eps_f,
            "no_harm": bool(ok),
            "learned_gains": {k: round(v, 3) for k, v in gains_l.items()},
        })
        print(f"  seed={seed}: guard learned={tl:.2f} (goals={n_goals_l}/"
              f"{n_eps_l}) frozen={tf:.2f} (goals={n_goals_f}/{n_eps_f}) "
              f"{'NO-HARM' if ok else 'HARM'} gains="
              f"{ {k: round(v, 2) for k, v in gains_l.items()} }")

    # bit-identity: guard must be inert on delayed_reward (shaping exists)
    bitid = {}
    ref = {}
    with open(os.path.join(
            FLESH, "var", "ecr-detector-delayed_reward-results.json")) as f:
        staged = json.load(f)
    for ps in staged["summary"]["per_seed"]:
        ref[ps["seed"]] = ps["learned_total"]
    for seed in BITID_SEEDS:
        tl, _, _, _ = probe(DelayedReward, seed, False, 8,
                            {"branch_a", "branch_b"}, False)
        match = abs(tl - ref[seed]) < 1e-12
        bitid[str(seed)] = {"guard_total": tl, "d3_total": ref[seed],
                            "bit_identical": bool(match)}
        print(f"  bit-identity seed={seed}: guard={tl:.6f} vs "
              f"D3={ref[seed]:.6f} {'MATCH' if match else 'MISMATCH'}")
        assert match, f"bit-identity FAIL on seed {seed}"

    verdict = "NO-HARM" if no_harm == NEED else "HARM-PERSISTS"
    print(f"  -> {verdict} ({no_harm}/{NEED})")
    config = {
        "env": "grid_world v1.0.0",
        "episodes_per_arm": 8, "max_steps": 100,
        "branch_channels": ["east", "north", "south", "west"],
        "specialists": "neutral (K5 P1 identical)",
        "capacity": 4, "gain_lr": GAIN_LR, "theta": THETA,
        "gain_cap": 2.0, "floor": 0.01,
        "arbitrator": "AdaptiveTerminalECR detector=relmax "
                      "progress_guard=True (EXP-FP-0126 module, additive "
                      "extension; canonical files UNTOUCHED)",
        "detector_params": {"FLOOR": 0.1, "REL": 0.5},
        "gate": "learned_goal_episodes >= frozen_goal_episodes on 4/4 "
                "(no-harm)",
        "seeds": list(SEEDS),
        "bit_identity_seeds": list(BITID_SEEDS),
    }
    result = {
        "experiment_id": "EXP-FP-0127",
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": SEEDS[0],
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "summary": {
            "per_seed": per_seed,
            "no_harm_count": no_harm, "need": NEED,
            "verdict": verdict,
            "bit_identity": bitid,
        },
    }
    path = os.path.join(FLESH, "var", "ecr-guard-grid_world-results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("staged results:", path)

    # G2 determinism spot-check.
    tl2, _, _, _ = probe(GridWorld, SEEDS[0], False, 8,
                         {"north", "south", "east", "west"}, True)
    tl1 = per_seed[0]["learned_total"]
    assert abs(tl2 - tl1) < 1e-12, f"G2 FAIL: {tl2} != {tl1}"
    print(f"G2: determinism spot-check PASS (seed {SEEDS[0]} "
          f"learned={tl1:.6f} exact)")


if __name__ == "__main__":
    main()
