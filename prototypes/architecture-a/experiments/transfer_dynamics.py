"""EXP-FP-0090 — transfer of LEARNING DYNAMICS (follow-up to TRANSFER-K4-RESTORE).

BACKGROUND (H0, TRANSFER-K4-RESTORE, 4/4): learned K4 gains do NOT transfer
as endpoint state. Transferred (frozen) gains help phase 1 (R~1.7) but hurt
phase 0 (T~18-26 vs F~98-104), net ~1.0. The K4 win is a PROCESS (online
gain-adaptation trajectory through the broadcast feedback loop), not a STATE.

THIS EXPERIMENT: test whether the learning DYNAMICS transfer, not the
endpoint. "Learning dynamics" = the adaptation hyperparameters that govern
the gain-adaptation process, meta-learned (by deterministic coordinate
descent) on a SOURCE task, then transferred into a fresh loop on a SHIFTED
task:

  SOURCE: canonical changing_rule, 10 episodes x 40 steps = 400 ticks
          (phase 0 -> phase 1 flip mid-run, cue-indexed adapter,
          theta=0.45, capacity=3 — the K4 canonical protocol).
  TARGET: post-flip changing_rule — 5 throwaway resets advance the env to
          episode_idx 5, so all 5 real episodes (200 ticks) sit in phase 1
          with the shifted (phase-1) rule from tick 0. The loop starts with
          gains = 1.0 and must RE-ADAPT to the new rule.

ARMS (per test seed, gains always start at 1.0, online learning unless noted):
  D  default dynamics  (gain_lr=0.15, z_alpha=0.01, initial_var=1.0,
      gain_cap=2.0, reward_baseline=0.5) — the fresh-learner control.
  X  TRANSFERRED dynamics (coordinate-descent-tuned on the source task,
      tuning seeds {97001,97002,97003}, disjoint from test seeds).
  E  H0 reference: default dynamics, FROZEN endpoint gains trained on the
      PRE-SHIFT world (phase-0-only source run, 200 ticks) — the
      TRANSFER-K4-RESTORE mechanism reproduced on the shifted task.

METRIC (preregistered): re-adaptation SPEED = time-to-criterion (TTC):
first 1-indexed tick t with mean(rewards[t-40:t]) >= 0.75 over the 200-tick
target run; None if never reached. Endpoint reward is NOT the metric
(H0 already closed that question).

GATES (preregistered):
  G0  fidelity: DynamicArbitrator(baseline=0.5) is bit-identical to the
      parent WorkspaceTick path on 400 ticks (rewards exactly equal).
  G1  tuning sanity: tuned config differs from default in >=1 parameter
      AND is no worse than default on the tuning seeds
      (mean_total(tuned) >= mean_total(default)). Else UNINTERPRETABLE.
  G2  transfer: (TTC_D - TTC_X)/TTC_D >= 0.30 on >= 3/4 test seeds,
      over seeds where D reaches criterion. >=2 uninformative seeds ->
      UNINTERPRETABLE.
  G3  no-harm: X reaches criterion on >= as many seeds as D.
  G4  H0 reference (reported): TTC_E is None on >= 3/4 seeds — the frozen
      pre-shift endpoint cannot re-adapt (void if phase-0 rule ==
      phase-1 rule, checked from the env's own phase-seeded rule).

Determinism: all RNG via env_interface.derive_seed / new_rng (seeded).
No random.* in measurement. Stdlib only.

Additive: this file is NEW. K4 originals (tick.py, attention.py,
consumers.py, broadcast.py, bids.py) are NOT modified. reward_baseline is
parameterized via DynamicArbitrator (arbitrator_cls hook, already supported
by WorkspaceTick); the gain update still arrives ONLY through the broadcast
feedback envelope — the sole-path discipline is intact.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                        # architecture-a
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "experiments", "envs"))

from tick import WorkspaceTick                                    # noqa: E402
from attention import AttentionArbitrator                        # noqa: E402
from env_interface import (CONTRACT_VERSION, config_hash,        # noqa: E402
                           derive_seed, new_rng)
from changing_rule import ChangingRule                           # noqa: E402
from transfer_k4_restore import (cue_indexed_specialists,         # noqa: E402
                                 ChannelEnv, CHANNELS)

EXPERIMENT_ID = "EXP-FP-0090"

# ---------------------------------------------------------------- config
DEFAULT_DYNAMICS = {"gain_lr": 0.15, "z_alpha": 0.01, "initial_var": 1.0,
                    "gain_cap": 2.0, "reward_baseline": 0.5}
PARAM_ORDER = ["gain_lr", "z_alpha", "initial_var", "gain_cap",
               "reward_baseline"]
TUNE_GRID = {"gain_lr": [0.05, 0.30],
             "z_alpha": [0.005, 0.05],
             "initial_var": [0.5, 2.0],
             "gain_cap": [1.5, 3.0],
             "reward_baseline": [0.3, 0.7]}
TUNING_SEEDS = (97001, 97002, 97003)
TEST_SEEDS = (97011, 97012, 97013, 97014)
PILOT_SEEDS = (97021, 97022)          # design calibration only, never reused
SOURCE_EPISODES = 10                  # 400 ticks, phase 0 -> phase 1
TARGET_EPISODES = 5                   # 200 ticks, phase 1 only (post-flip)
THROWAWAY_RESETS = 5                  # advance episode_idx  -1 -> 4
STEPS = 40
THETA = 0.45
CAPACITY = 3
CRITERION = 0.75
WINDOW = 40
SPEEDUP_MARGIN = 0.30
NEED = 3


class DynamicArbitrator(AttentionArbitrator):
    """Additive dynamics hook: reward_baseline becomes a property of the
    adaptation dynamics, owned by the arbitrator.

    The gain update STILL arrives only via the broadcast feedback envelope
    (consumers.py AttentionUpdateConsumer -> update_gains): the sole path is
    intact. The payload's reward_baseline field is superseded by the
    configured dynamics value (documented override, not a side channel).
    With reward_baseline=0.5 this is bit-identical to the parent path (G0).
    """

    def __init__(self, *args, reward_baseline=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_baseline = float(reward_baseline)

    def update_gains(self, utility, reward_baseline=0.0):
        return super().update_gains(utility,
                                    reward_baseline=self.reward_baseline)


def make_tick(dynamics, frozen=False, preload_gains=None):
    wk = WorkspaceTick(
        CHANNELS, cue_indexed_specialists(CHANNELS), capacity=CAPACITY,
        gain_lr=dynamics["gain_lr"], frozen_gains=frozen,
        ignition_kwargs={"theta": THETA},
        alpha=dynamics["z_alpha"], initial_var=dynamics["initial_var"],
        arbitrator_cls=DynamicArbitrator,
        arbitrator_kwargs={"gain_cap": dynamics["gain_cap"],
                           "reward_baseline": dynamics["reward_baseline"]})
    if preload_gains is not None:
        for c, g in preload_gains.items():
            wk.arbitrator.gains[c] = float(g)
        wk.arbitrator.freeze()
    return wk


def drive(wk, env, n_episodes, seed, start_run_index=0):
    """Drive n_episodes of the closed loop; returns the rewards list."""
    rewards = []
    for run_index in range(start_run_index, start_run_index + n_episodes):
        obs = env.reset(derive_seed(seed, run_index, "env"))
        chan_env = ChannelEnv(env)
        for _ in range(STEPS):
            wk.step({"cue": int(obs["cue"]),
                     "last_reward": float(obs["last_reward"]),
                     "last_action": int(obs["last_action"])}, chan_env)
            rewards.append(wk.rewards[-1])
            obs = chan_env.last_obs2
            if chan_env.last_done:
                break
    return rewards


def source_run(seed, dynamics, episodes=SOURCE_EPISODES):
    """Canonical source protocol: 10 episodes, phase 0 -> phase 1."""
    wk = make_tick(dynamics)
    env = ChangingRule()
    rewards = drive(wk, env, episodes, seed)
    return {"total": float(sum(rewards)), "rewards": rewards,
            "gains": dict(wk.arbitrator.gains),
            "n_gain_updates": len(wk.arbitrator.gain_history)}


def target_run(seed, dynamics, frozen_gains=None):
    """Shifted target: post-flip changing_rule.

    5 throwaway resets advance episode_idx to 4; the 5 real episodes then
    run at episode_idx 5..9 = phase 1 throughout (the shifted rule from
    tick 0). The loop starts with gains = 1.0 and must re-adapt.
    """
    wk = make_tick(dynamics, frozen=(frozen_gains is not None),
                   preload_gains=frozen_gains)
    env = ChangingRule()
    for i in range(THROWAWAY_RESETS):
        env.reset(derive_seed(0, i, "throwaway"))
    rewards = drive(wk, env, TARGET_EPISODES, seed)
    return {"total": float(sum(rewards)), "rewards": rewards,
            "ttc": time_to_criterion(rewards)}


def time_to_criterion(rewards, window=WINDOW, criterion=CRITERION):
    """First 1-indexed tick t with mean(rewards[t-window:t]) >= criterion."""
    for t in range(window, len(rewards) + 1):
        if sum(rewards[t - window:t]) / window >= criterion:
            return t
    return None


def phase_rule(phase):
    """The env's own phase-seeded rule (analysis labeling only; the agent
    never sees it)."""
    rule_rng = new_rng((phase * 7919 + 0x5EED) % (2 ** 31))
    return [rule_rng.randrange(2) for _ in range(ChangingRule.N_CUES)]


def mean_total(dynamics, seeds, episodes=SOURCE_EPISODES):
    return sum(source_run(s, dynamics, episodes)["total"]
               for s in seeds) / len(seeds)


def tune_dynamics():
    """Deterministic coordinate descent on the source task (meta-learning
    the dynamics). Two passes max over PARAM_ORDER; a candidate replaces
    the incumbent only on STRICT improvement of tuning-seed mean total;
    ties keep the incumbent (parsimony toward the default)."""
    cfg = dict(DEFAULT_DYNAMICS)
    trace = []
    for pass_no in (1, 2):
        changed = False
        for param in PARAM_ORDER:
            base = mean_total(cfg, TUNING_SEEDS)
            best_val, best_mean = cfg[param], base
            for cand in TUNE_GRID[param]:
                trial = dict(cfg)
                trial[param] = cand
                m = mean_total(trial, TUNING_SEEDS)
                trace.append({"pass": pass_no, "param": param,
                              "candidate": cand, "mean_total": m,
                              "incumbent": base})
                if m > best_mean:
                    best_val, best_mean = cand, m
            if best_val != cfg[param]:
                trace.append({"pass": pass_no, "param": param,
                              "ADOPTED": best_val, "from": cfg[param],
                              "to_mean": best_mean, "from_mean": base})
                cfg[param] = best_val
                changed = True
        if not changed:
            break
    return cfg, trace


def bit_identity_check(seed=PILOT_SEEDS[0]):
    """G0: DynamicArbitrator(baseline=0.5) bit-identical to the parent path."""
    wk_new = make_tick(DEFAULT_DYNAMICS)
    wk_old = WorkspaceTick(CHANNELS, cue_indexed_specialists(CHANNELS),
                           capacity=CAPACITY, gain_lr=0.15,
                           frozen_gains=False,
                           ignition_kwargs={"theta": THETA})
    env1, env2 = ChangingRule(), ChangingRule()
    r_new = drive(wk_new, env1, SOURCE_EPISODES, seed)
    r_old = drive(wk_old, env2, SOURCE_EPISODES, seed)
    return r_new == r_old, len(r_new)


def main():
    t_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = {"experiment_id": EXPERIMENT_ID, "t_wall": t_wall}

    # ---- G0 fidelity -------------------------------------------------
    identical, n = bit_identity_check()
    out["G0_bit_identity"] = {"identical": identical, "ticks": n}
    assert identical, "G0 FAILED: DynamicArbitrator path diverged from parent"

    # ---- tune dynamics on the source task (deterministic) ------------
    tuned, trace = tune_dynamics()
    default_mean = mean_total(DEFAULT_DYNAMICS, TUNING_SEEDS)
    tuned_mean = mean_total(tuned, TUNING_SEEDS)
    g1_distinct = any(tuned[p] != DEFAULT_DYNAMICS[p] for p in PARAM_ORDER)
    g1_no_worse = tuned_mean >= default_mean
    out["tuning"] = {"tuning_seeds": list(TUNING_SEEDS),
                     "default_dynamics": DEFAULT_DYNAMICS,
                     "tuned_dynamics": tuned,
                     "default_mean_total": default_mean,
                     "tuned_mean_total": tuned_mean,
                     "distinct": g1_distinct, "no_worse": g1_no_worse,
                     "trace": trace}
    g1 = g1_distinct and g1_no_worse
    out["G1_tuning_sanity"] = g1

    # ---- test on 4 fresh seeds --------------------------------------
    rule0, rule1 = phase_rule(0), phase_rule(1)
    out["task_shift"] = {"phase0_rule": rule0, "phase1_rule": rule1,
                         "rules_differ": rule0 != rule1}
    per_seed = []
    for seed in TEST_SEEDS:
        # Arm E endpoint: train on the PRE-SHIFT world (phase 0 only).
        src0 = source_run(seed, DEFAULT_DYNAMICS, episodes=5)
        arm_d = target_run(seed, DEFAULT_DYNAMICS)
        arm_x = target_run(seed, tuned)
        arm_e = target_run(seed, DEFAULT_DYNAMICS,
                           frozen_gains=src0["gains"])
        rec = {"seed": seed,
               "arm_D": {"ttc": arm_d["ttc"], "total": arm_d["total"]},
               "arm_X": {"ttc": arm_x["ttc"], "total": arm_x["total"]},
               "arm_E": {"ttc": arm_e["ttc"], "total": arm_e["total"]},
               "endpoint_gains_phase0": src0["gains"]}
        per_seed.append(rec)
    out["per_seed"] = per_seed

    # ---- gates --------------------------------------------------------
    informative = [s for s in per_seed if s["arm_D"]["ttc"] is not None]
    uninformative = len(per_seed) - len(informative)
    wins = sum(1 for s in informative
               if s["arm_X"]["ttc"] is not None
               and (s["arm_D"]["ttc"] - s["arm_X"]["ttc"])
               / s["arm_D"]["ttc"] >= SPEEDUP_MARGIN)
    reach_d = sum(1 for s in per_seed if s["arm_D"]["ttc"] is not None)
    reach_x = sum(1 for s in per_seed if s["arm_X"]["ttc"] is not None)
    reach_e = sum(1 for s in per_seed if s["arm_E"]["ttc"] is not None)
    g2 = (wins >= NEED) if (g1 and uninformative < 2) else None
    g3 = reach_x >= reach_d
    g4 = (reach_e <= len(per_seed) - NEED) if (rule0 != rule1) else None
    out["gates"] = {
        "G2_transfer_pass": g2, "G2_wins": wins,
        "G2_informative_seeds": len(informative),
        "G2_uninformative_seeds": uninformative,
        "G3_no_harm_pass": g3, "reach_D": reach_d, "reach_X": reach_x,
        "G4_H0_reference": g4, "reach_E": reach_e,
        "rules_differ": rule0 != rule1}
    if not g1:
        verdict = ("UNINTERPRETABLE — G1 tuning sanity failed: the source "
                   "task produced no distinct, no-worse dynamics to "
                   "transfer. The transfer question has no lever.")
    elif uninformative >= 2:
        verdict = ("UNINTERPRETABLE — default dynamics failed to reach "
                   "criterion on >=2 seeds; speedup is undefined.")
    elif g2:
        verdict = ("H1 SUPPORTED — transferred learning dynamics re-adapt "
                   ">=30% faster on >=3/4 seeds: the adaptation PROCESS "
                   "carries task-family knowledge, not just the endpoint.")
    else:
        verdict = ("H0 — learning dynamics do NOT transfer: source-tuned "
                   "dynamics re-adapt no faster than default dynamics on "
                   "the shifted task (wins %d/4, need %d)." % (wins, NEED))
    out["verdict"] = verdict

    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "G0": out["G0_bit_identity"], "G1": g1,
        "tuned_dynamics": tuned,
        "tuning_means": {"default": round(default_mean, 3),
                         "tuned": round(tuned_mean, 3)},
        "per_seed_ttc": {s["seed"]: {"D": s["arm_D"]["ttc"],
                                     "X": s["arm_X"]["ttc"],
                                     "E": s["arm_E"]["ttc"]} for s in per_seed},
        "G2": g2, "G2_wins": wins, "G3": g3, "G4": g4,
        "verdict": verdict}, indent=1))
    state_path = os.path.join(HERE, "..", "receipts", "transfer",
                              "EXP-FP-0090-state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


if __name__ == "__main__":
    main()
