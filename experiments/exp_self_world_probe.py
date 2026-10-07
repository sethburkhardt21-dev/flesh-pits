"""EXP-SW-01 — self/world distinction probe (§9 item 12).

Preregistration: experiments/preregistration_SELF_WORLD.json (written BEFORE
any run). This script implements exactly that procedure.

Two arms (asymmetric by design — the architectures compute different things):
  B (arch_b): open-loop transition training (learn_transition, K3C-style),
      intact vs frozen (paired streams, paired init seeds). Held-out eval via
      model.predict_next; carrying metric D = mean|e_ball| - mean|e_hand|.
  A (WorkspaceTick): closed-loop via a scalar-reward adapter on self_world;
      change-detector specialists; carrying metrics Delta_bid, Delta_ign.

Ground-truth cause labels come from env info — read by THIS SCRIPT for
scoring only. Neither agent ever sees info (contract v1.0.0 §5).

Usage:
    python3 exp_self_world_probe.py --arch b --out ../receipts
    python3 exp_self_world_probe.py --arch a --out ../receipts

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
sys.path.insert(0, os.path.join(HERE, "..", "prototypes", "architecture-a"))

from env_interface import (  # noqa: E402
    derive_seed, new_rng, obs_to_vector, config_hash,
)
from self_world import SelfWorld  # noqa: E402
from harness import write_receipt  # noqa: E402

SEEDS = [75101, 75102, 75103, 75104]
INIT_SEED_BASE = 75000
N_TRAIN_EPISODES = 10
N_EVAL_EPISODES = 5
A_EPISODES = 4


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def space_of(env):
    return env.observation_space()


def policy_actions(seed, episode, n_steps):
    """Deterministic balanced-ish action stream, shared by paired arms."""
    rng = new_rng(derive_seed(seed, episode, "policy"))
    return [rng.randint(0, 1) for _ in range(n_steps)]


def check_determinism():
    """G0b: same (seed, action sequence) -> byte-identical trajectory."""
    acts = policy_actions(99901, 0, SelfWorld.MAX_STEPS)
    trajs = []
    for _ in range(2):
        env = SelfWorld()
        obs = env.reset(99901)
        traj = [obs["hand"], obs["ball"]]
        for a in acts:
            obs, _, _, _ = env.step(a)
            traj += [obs["hand"], obs["ball"]]
        trajs.append(traj)
    assert trajs[0] == trajs[1], "G0b FAILED: env not deterministic"
    return True


# --------------------------------------------------------------------------
# B arm
# --------------------------------------------------------------------------

def run_b_seed(seed):
    from agent import ArchB  # local import: keeps --arch a runnable alone
    env = SelfWorld()
    space = space_of(env)
    init_seed = INIT_SEED_BASE + (SEEDS.index(seed) + 1)
    intact = ArchB(space, 2, env_name="self_world", affect="none",
                   seed=init_seed)
    frozen = ArchB(space, 2, env_name="self_world", affect="none",
                   seed=init_seed, frozen=True)

    def train(agent):
        for ep in range(N_TRAIN_EPISODES):
            acts = policy_actions(seed, ep, SelfWorld.MAX_STEPS)
            obs = env.reset(derive_seed(seed, ep, "env"))
            for a in acts:
                obs2, r, done, info = env.step(a)
                agent.learn_transition(obs_to_vector(obs, space), a,
                                       obs_to_vector(obs2, space), r)
                assert info["cause"] == {"hand": "self", "ball": "world"}
                obs = obs2
                if done:
                    break

    train(intact)
    train(frozen)

    def held_out(agent):
        e_hand, e_ball = [], []
        n = 0
        for ep in range(N_EVAL_EPISODES):
            acts = policy_actions(seed, 100 + ep, SelfWorld.MAX_STEPS)
            obs = env.reset(derive_seed(seed, 100 + ep, "env"))
            for a in acts:
                obs2, r, done, info = env.step(a)
                v, v2 = obs_to_vector(obs, space), obs_to_vector(obs2, space)
                ctx = agent.context_key(v)
                xhat = agent.model.predict_next(v, a, ctx)
                e_hand.append(abs(v2[0] - xhat[0]))
                e_ball.append(abs(v2[1] - xhat[1]))
                n += 1
                obs = obs2
                if done:
                    break
        assert n > 0, "G0a: held-out transition count == 0"
        return e_hand, e_ball

    ih, ib = held_out(intact)
    fh, fb = held_out(frozen)
    d_intact = statistics.fmean(ib) - statistics.fmean(ih)
    d_frozen = statistics.fmean(fb) - statistics.fmean(fh)
    return {
        "seed": seed,
        "n_heldout": len(ih),
        "intact_mean_e_hand": statistics.fmean(ih),
        "intact_mean_e_ball": statistics.fmean(ib),
        "frozen_mean_e_hand": statistics.fmean(fh),
        "frozen_mean_e_ball": statistics.fmean(fb),
        "D_intact": d_intact,
        "D_frozen": d_frozen,
    }


def verdict_b(per_seed):
    # AMENDED (preregistration amendment, pre-run): carrying metric is the
    # paired bias-corrected gap D_adj = D_intact - D_frozen, both arms
    # evaluated on the IDENTICAL held-out stream. The frozen arm is the
    # paired baseline inside the metric, not a separate bound.
    ds = [p["D_intact"] - p["D_frozen"] for p in per_seed]
    for p, d in zip(per_seed, ds):
        p["D_adj"] = d
    mean_d = statistics.fmean(ds)
    pos = sum(1 for x in ds if x > 0)
    ok = mean_d > 0.05 and pos >= 3
    return {
        "seed_mean_D_adj": mean_d,
        "seed_mean_D_intact": statistics.fmean([p["D_intact"] for p in per_seed]),
        "seed_mean_D_frozen": statistics.fmean([p["D_frozen"] for p in per_seed]),
        "seeds_positive_D_adj": pos,
        "per_seed": per_seed,
        "verdict": ("DISTINCTION DETECTED (PASS)" if ok
                    else "NO DISTINCTION (null holds)"),
        "rule_fired": ok,
    }


# --------------------------------------------------------------------------
# A arm
# --------------------------------------------------------------------------

def make_change_specialists(channels):
    """Specialists: stimulus = |channel_t - channel_{t-1}| (change detector).

    Payloads are neutral. No action information reaches the specialists —
    that is the architectural fact under test.
    """
    last = {}

    def make_fn(c):
        def fn(obs, tick):
            v = float(obs[c])
            prev = last.get(c, v)
            last[c] = v
            d = abs(v - prev)
            return d, {"summary": f"delta:{c}", "delta": d}
        return fn

    return [(c, make_fn(c)) for c in channels]


class RewardAdapter:
    """Wraps a v1 Environment for A's tick, which expects env.step(action)
    to return a scalar reward. Caches the full transition for the loop."""

    def __init__(self, env, action_map):
        self.env = env
        self.action_map = dict(action_map)
        self.last = None

    def step(self, action):
        obs2, reward, done, info = self.env.step(self.action_map[action])
        self.last = (obs2, reward, done, info)
        return reward


def run_a_seed(seed):
    from tick import WorkspaceTick  # local import
    channels = ["hand", "ball"]
    tick = WorkspaceTick(channels, make_change_specialists(channels),
                         capacity=3, gain_lr=0.05)
    env = SelfWorld()
    adapter = RewardAdapter(env, {"hand": 0, "ball": 1})

    bids = {"hand": [], "ball": []}
    ignited = {"hand": [], "ball": []}
    strengths = {"hand": [], "ball": []}
    n_ticks = 0
    for ep in range(A_EPISODES):
        obs = env.reset(derive_seed(seed, ep, "env"))
        for _ in range(SelfWorld.MAX_STEPS):
            trace = tick.step(obs, adapter)
            obs2, _, done, _ = adapter.last
            # item -> channel: the tick admits channels in self.channels
            # order (verified against tick.step source); admissions are
            # recorded in that same order.
            chan_of = {}
            for rec, c in zip(trace["admissions"], channels):
                chan_of[rec["item_id"]] = c
            comp = trace["arbitration"]["competed_bids"]
            for c in channels:
                bids[c].append(comp[c])
            for ig in trace["ignitions"]:
                c = chan_of[ig["item_id"]]
                ignited[c].append(1.0 if ig["ignited"] else 0.0)
                strengths[c].append(ig["strength"])
            n_ticks += 1
            obs = obs2
            if done:
                break

    def mean(xs):
        return statistics.fmean(xs) if xs else 0.0

    d_bid = mean(bids["hand"]) - mean(bids["ball"])
    r_hand = mean(ignited["hand"])
    r_ball = mean(ignited["ball"])
    gains = tick.stats()["gains"]
    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "mean_bid_hand": mean(bids["hand"]),
        "mean_bid_ball": mean(bids["ball"]),
        "Delta_bid": d_bid,
        "ign_rate_hand": r_hand,
        "ign_rate_ball": r_ball,
        "Delta_ign": r_hand - r_ball,
        "mean_strength_hand": mean(strengths["hand"]),
        "mean_strength_ball": mean(strengths["ball"]),
        "gain_hand": gains["hand"],
        "gain_ball": gains["ball"],
        "Delta_gain_diagnostic": gains["hand"] - gains["ball"],
    }


def verdict_a(per_seed):
    db = [p["Delta_bid"] for p in per_seed]
    di = [p["Delta_ign"] for p in per_seed]
    m_db, m_di = statistics.fmean(db), statistics.fmean(di)
    s_db = sum(1 for x in db if (x > 0) == (m_db > 0) and x != 0.0)
    s_di = sum(1 for x in di if (x > 0) == (m_di > 0) and x != 0.0)
    hit = ((abs(m_db) > 0.05 and s_db >= 3)
           or (abs(m_di) > 0.05 and s_di >= 3))
    return {
        "seed_mean_Delta_bid": m_db,
        "seed_mean_Delta_ign": m_di,
        "per_seed": per_seed,
        "verdict": ("DISTINCTION DETECTED" if hit
                    else "NO DISTINCTION (null holds — expected negative)"),
        "rule_fired": hit,
    }

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

PREREG = {
    "B": {
        "hypothesis": ("B's action-conditioned generative model predicts "
                       "self-caused observation changes better than "
                       "world-caused ones: mean|e|_ball > mean|e|_hand on "
                       "held-out transitions."),
        "null": ("D ~= 0 intact, or the frozen arm shows the same gap "
                 "(architectural bias, not learned attribution)."),
        "metric": ("D_adj = D_intact - D_frozen per seed (AMENDED pre-run: paired "
                   "bias-corrected gap, identical held-out stream for both "
                   "arms); PASS iff seed-mean D_adj > +0.05 AND D_adj > 0 on "
                   ">= 3/4 seeds."),
        "baseline": "frozen ArchB (frozen=True), identical streams + init seeds.",
    },
    "A": {
        "hypothesis": ("No internal variable of A (competed bid, ignition "
                       "rate/strength) systematically distinguishes "
                       "self-caused from world-caused changes under matched "
                       "statistics — expected negative."),
        "null": ("The null IS the expectation for A. A gap would mean A's "
                 "machinery carries a distinction its architecture cannot "
                 "explain."),
        "metric": ("Delta_bid = mean bid_hand - bid_ball; Delta_ign = "
                   "ign_rate_hand - ign_rate_ball; DISTINCTION iff "
                   "|seed-mean| > 0.05 on either with consistent sign "
                   ">= 3/4 seeds."),
        "baseline": ("no baseline arm; instrument validated by unit test "
                     "(chain discriminates unequal stimuli by construction)."),
    },
}

CONDITIONS = (
    "self_world v1.0.0; STEP=0.2; MAX_STEPS=60; reward 0.0; contract 1.0.0; "
    "cause labels from info used for SCORING ONLY, never by agents. "
    "B: ArchB affect='none', learn_transition training (10 eps x 60), "
    "held-out predict_next eval (5 eps x 60); init seed 75001..75004 paired. "
    "A: WorkspaceTick channels ['hand','ball'], change-detector specialists, "
    "capacity=3, gain_lr=0.05, closed-loop via scalar-reward adapter "
    "(4 eps x 60 ticks), action map {'hand':0,'ball':1}."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["a", "b"], required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "receipts"))
    args = ap.parse_args()

    check_determinism()  # G0b, fail-closed before any measurement

    arch = args.arch.upper()
    per_seed = [run_b_seed(s) if arch == "B" else run_a_seed(s)
                for s in SEEDS]
    res = verdict_b(per_seed) if arch == "B" else verdict_a(per_seed)

    exp_id = f"EXP-SW-01-{arch}"
    config = {
        "experiment_id": exp_id,
        "arch": arch,
        "env": "self_world",
        "env_version": SelfWorld.VERSION,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "preregistration": "experiments/preregistration_SELF_WORLD.json",
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
    pre = PREREG[arch]
    path = write_receipt(
        result, args.out,
        hypothesis=pre["hypothesis"], null=pre["null"],
        preregistered_metric=pre["metric"], baseline=pre["baseline"],
        conditions=CONDITIONS,
        interpretation=res["verdict"], limitations="")
    print(json.dumps({"experiment_id": exp_id, "receipt": path,
                      "verdict": res["verdict"], "summary": res},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
