"""EXP-SW-02-B — closed-loop extension of the self/world distinction probe.

Preregistration: experiments/preregistration_SELF_WORLD_CL.json (written
BEFORE any run). This script implements exactly that procedure.

Question: does the self/world distinction EXP-SW-01 detected in arch-B's
open-loop prediction error do WORK during closed-loop control, or is it
only visible in prediction error?

Design (B-only, preregistered scope):
  intact   : ArchB trained open-loop on self_world via learn_transition
             (identical procedure to EXP-SW-01 B), then FROZEN
             (base_etas zeroed — no learning during control).
  shuffle  : identical streams, init seeds, and training budget, but
             learn_transition receives shuffled action labels
             (derive_seed(s, ep, "shuffle") stream) — learns transition
             marginals but cannot learn action-conditioning. Same freeze.
  frozen   : ArchB(frozen=True), no training (no-learning control).
  controller (all arms, same code): greedy one-step planner over
             model.predict_next — choose a minimizing |xhat[hand] - 0.5|;
             ties keep the previous action. No act()/update() path is used
             in control; the model's action-conditioned predictions are
             what drives action selection.
  env      : self_world_cl v1.0.0 (additive subclass of self_world;
             reward = -(hand - 0.5)^2). 5 episodes x 60 steps per arm.

Carrying metric: R_gap_attrib = return_intact - return_shuffle per seed.
ATTRIBUTION ADVANTAGE (PASS) iff seed-mean R_gap_attrib > +1.0 AND
R_gap_attrib > 0 on >= 3/4 seeds.

Ground-truth cause labels come from env info — read by THIS SCRIPT for
scoring only (online D diagnostic). Neither agent ever sees info
(contract v1.0.0 §5).

Usage:
    python3 exp_self_world_closedloop.py --out ../receipts

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
sys.path.insert(0, os.path.join(HERE, "..", "prototypes", "architecture-b"))

from env_interface import (  # noqa: E402
    derive_seed, new_rng, obs_to_vector, config_hash,
)
from self_world import SelfWorld  # noqa: E402
from self_world_cl import SelfWorldCL  # noqa: E402
from harness import write_receipt  # noqa: E402

SEEDS = [75201, 75202, 75203, 75204]
INIT_SEED_BASE = 76000
N_TRAIN_EPISODES = 10
N_CONTROL_EPISODES = 5
HAND_IDX = 0  # obs_to_vector order follows observation_space insertion


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def policy_actions(seed, episode, n_steps):
    """Deterministic balanced-ish action stream (same as EXP-SW-01)."""
    rng = new_rng(derive_seed(seed, episode, "policy"))
    return [rng.randint(0, 1) for _ in range(n_steps)]


def shuffle_actions(seed, episode, n_steps):
    """Independent action-label stream for the attribution lesion."""
    rng = new_rng(derive_seed(seed, episode, "shuffle"))
    return [rng.randint(0, 1) for _ in range(n_steps)]


def check_determinism():
    """G0b: same (seed, action sequence) -> byte-identical trajectory."""
    acts = policy_actions(99902, 0, SelfWorld.MAX_STEPS)
    for cls in (SelfWorld, SelfWorldCL):
        trajs = []
        for _ in range(2):
            env = cls()
            obs = env.reset(99902)
            traj = [obs["hand"], obs["ball"]]
            for a in acts:
                obs, r, _, _ = env.step(a)
                traj += [obs["hand"], obs["ball"], r]
            trajs.append(traj)
        assert trajs[0] == trajs[1], \
            f"G0b FAILED: {cls.NAME} not deterministic"
    return True


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_arm(agent, seed, shuffle_labels):
    """Offline open-loop transition training (EXP-SW-01 B procedure)."""
    env = SelfWorld()
    space = env.observation_space()
    ep_hand_err = []
    for ep in range(N_TRAIN_EPISODES):
        acts = policy_actions(seed, ep, SelfWorld.MAX_STEPS)
        labels = (shuffle_actions(seed, ep, SelfWorld.MAX_STEPS)
                  if shuffle_labels else acts)
        obs = env.reset(derive_seed(seed, ep, "env"))
        errs = []
        for a, lab in zip(acts, labels):
            obs2, r, done, info = env.step(a)
            v = obs_to_vector(obs, space)
            v2 = obs_to_vector(obs2, space)
            e0 = agent.learn_transition(v, lab, v2, r)
            errs.append(abs(v2[HAND_IDX]
                            - agent.model.predict_next(v, lab,
                                                       agent.context_key(v))
                            [HAND_IDX]))
            assert info["cause"] == {"hand": "self", "ball": "world"}
            obs = obs2
            if done:
                break
        ep_hand_err.append(statistics.fmean(errs) if errs else 0.0)
    first = statistics.fmean(ep_hand_err[:2])
    last = statistics.fmean(ep_hand_err[-2:])
    return {"ep_hand_err_first2": first, "ep_hand_err_last2": last,
            "g0c_learning_sanity": last < first}


# --------------------------------------------------------------------------
# closed-loop control
# --------------------------------------------------------------------------

def greedy_action(agent, space, obs, last_action):
    """One-step greedy planner on model.predict_next (preregistered).

    Chooses a minimizing |xhat[hand] - TARGET|; ties keep last_action.
    """
    v = obs_to_vector(obs, space)
    ctx = agent.context_key(v)
    best, best_cost = last_action, None
    # last_action first with strict < : ties keep the previous action.
    for a in (last_action, 1 - last_action):
        xhat = agent.model.predict_next(v, a, ctx)
        cost = abs(xhat[HAND_IDX] - SelfWorldCL.TARGET)
        if best_cost is None or cost < best_cost:
            best, best_cost = a, cost
    return best


def run_control(agent, seed):
    """Closed-loop control on self_world_cl; model frozen (no learning).

    Returns (total_return, n_episodes, online D diagnostics).
    """
    env = SelfWorldCL()
    space = env.observation_space()
    total = 0.0
    n_ep = 0
    e_hand, e_ball = [], []
    for ep in range(N_CONTROL_EPISODES):
        obs = env.reset(derive_seed(seed, 200 + ep, "env"))
        last_action = 0
        for _ in range(SelfWorldCL.MAX_STEPS):
            a = greedy_action(agent, space, obs, last_action)
            last_action = a
            v = obs_to_vector(obs, space)
            ctx = agent.context_key(v)
            xhat = agent.model.predict_next(v, a, ctx)
            obs2, reward, done, info = env.step(a)
            v2 = obs_to_vector(obs2, space)
            e_hand.append(abs(v2[HAND_IDX] - xhat[HAND_IDX]))
            e_ball.append(abs(v2[1] - xhat[1]))
            total += reward
            assert info["cause"] == {"hand": "self", "ball": "world"}
            obs = obs2
            if done:
                break
        n_ep += 1
    return {
        "return": total,
        "n_control_episodes": n_ep,
        "online_mean_e_hand": statistics.fmean(e_hand),
        "online_mean_e_ball": statistics.fmean(e_ball),
        "online_D": statistics.fmean(e_ball) - statistics.fmean(e_hand),
    }


def freeze(agent):
    """Zero the learning rates actually consulted by observe() in control.

    The control loop never calls observe() anyway; this is defense in
    depth so no online weight change can re-learn the distinction.
    """
    agent.base_etas = (0.0, 0.0, 0.0, 0.0)
    agent.model.eta0 = 0.0
    agent.model.etaD = 0.0
    agent.model.eta_r = 0.0
    agent.model.etaR = 0.0


def run_seed(seed, init_seed=None):
    from agent import ArchB  # local import
    env = SelfWorld()
    space = env.observation_space()
    if init_seed is None:
        init_seed = INIT_SEED_BASE + (SEEDS.index(seed) + 1)

    intact = ArchB(space, 2, env_name="self_world", affect="none",
                   seed=init_seed)
    shuffle = ArchB(space, 2, env_name="self_world", affect="none",
                    seed=init_seed)
    frozen = ArchB(space, 2, env_name="self_world", affect="none",
                   seed=init_seed, frozen=True)

    intact_train = train_arm(intact, seed, shuffle_labels=False)
    shuffle_train = train_arm(shuffle, seed, shuffle_labels=True)
    # frozen arm: no training.

    freeze(intact)
    freeze(shuffle)
    freeze(frozen)

    ci = run_control(intact, seed)
    cs = run_control(shuffle, seed)
    cf = run_control(frozen, seed)
    assert ci["n_control_episodes"] > 0, "G0a: intact zero episodes"
    assert cs["n_control_episodes"] > 0, "G0a: shuffle zero episodes"
    assert cf["n_control_episodes"] > 0, "G0a: frozen zero episodes"

    return {
        "seed": seed,
        "return_intact": ci["return"],
        "return_shuffle": cs["return"],
        "return_frozen": cf["return"],
        "R_gap_attrib": ci["return"] - cs["return"],
        "R_gap_learn": ci["return"] - cf["return"],
        "online_D_intact": ci["online_D"],
        "online_D_shuffle": cs["online_D"],
        "online_mean_e_hand_intact": ci["online_mean_e_hand"],
        "online_mean_e_ball_intact": ci["online_mean_e_ball"],
        "g0c_intact": intact_train["g0c_learning_sanity"],
        "g0c_shuffle": shuffle_train["g0c_learning_sanity"],
    }


def verdict(per_seed):
    gaps = [p["R_gap_attrib"] for p in per_seed]
    mean_gap = statistics.fmean(gaps)
    pos = sum(1 for x in gaps if x > 0)
    ok = mean_gap > 1.0 and pos >= 3
    return {
        "seed_mean_R_gap_attrib": mean_gap,
        "seed_mean_R_gap_learn": statistics.fmean(
            [p["R_gap_learn"] for p in per_seed]),
        "seed_mean_return_intact": statistics.fmean(
            [p["return_intact"] for p in per_seed]),
        "seed_mean_return_shuffle": statistics.fmean(
            [p["return_shuffle"] for p in per_seed]),
        "seed_mean_return_frozen": statistics.fmean(
            [p["return_frozen"] for p in per_seed]),
        "seed_mean_online_D_intact": statistics.fmean(
            [p["online_D_intact"] for p in per_seed]),
        "seeds_positive_R_gap_attrib": pos,
        "g0c_flags": [p["g0c_intact"] for p in per_seed],
        "per_seed": per_seed,
        "verdict": ("ATTRIBUTION ADVANTAGE (PASS)" if ok
                    else "NO ATTRIBUTION ADVANTAGE (null holds)"),
        "rule_fired": ok,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

CONDITIONS = (
    "self_world v1.0.0 offline training (10 eps x 60 learn_transition); "
    "self_world_cl v1.0.0 control (reward = -(hand - 0.5)^2; 5 eps x 60, "
    "model frozen, base_etas zeroed); greedy one-step planner on "
    "model.predict_next minimizing |xhat[hand] - 0.5|, ties keep last "
    "action; arms: intact / action-shuffled (paired streams+init seeds "
    "76001..76004) / frozen; cause labels from info for SCORING ONLY, "
    "never by agents; contract 1.0.0."
)

PREREG = {
    "hypothesis": ("The self/world distinction does closed-loop WORK: the "
                   "action-conditioned model's greedy controller earns "
                   "higher return than the action-shuffled control."),
    "null": ("R_gap_attrib ~= 0: no control advantage from the "
             "action-conditioned attribution (measurement-only "
             "epiphenomenon), or the gap is generic learning."),
    "metric": ("R_gap_attrib = return_intact - return_shuffle per seed; "
               "ATTRIBUTION ADVANTAGE (PASS) iff seed-mean > +1.0 AND "
               "> 0 on >= 3/4 seeds."),
    "baseline": ("action-shuffled ArchB (attribution lesion); secondary: "
                 "frozen ArchB (no learning)."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "receipts"))
    ap.add_argument("--smoke", action="store_true",
                    help="throwaway-seed methods validation (99902); "
                         "no receipt, no experimental seeds touched")
    args = ap.parse_args()

    check_determinism()  # G0b, fail-closed before any measurement

    if args.smoke:
        p = run_seed(99902, init_seed=99901)
        print(json.dumps({"smoke_seed": 99902, "smoke_result": p},
                         indent=2, default=str))
        return

    per_seed = [run_seed(s) for s in SEEDS]
    res = verdict(per_seed)

    exp_id = "EXP-SW-02-B"
    config = {
        "experiment_id": exp_id,
        "arch": "B",
        "env_train": "self_world",
        "env_control": "self_world_cl",
        "env_control_version": SelfWorldCL.VERSION,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "preregistration": "experiments/preregistration_SELF_WORLD_CL.json",
    }
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = {
        "experiment_id": exp_id,
        "config": config,
        "config_hash": config_hash(config),
        "primary_seed": SEEDS,
        "started_utc": now,
        "episodes": per_seed,
        "summary": res,
    }
    path = write_receipt(
        result, args.out,
        hypothesis=PREREG["hypothesis"], null=PREREG["null"],
        preregistered_metric=PREREG["metric"], baseline=PREREG["baseline"],
        conditions=CONDITIONS,
        interpretation=res["verdict"],
        limitations="CONSCIOUSNESS: UNRESOLVED — a control advantage is a "
                    "mechanism, not a subject.")
    print(json.dumps({"experiment_id": exp_id, "receipt": path,
                      "verdict": res["verdict"], "summary": res},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
