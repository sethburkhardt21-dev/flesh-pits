"""Kill experiments K1-K5 + load-bearing claim tests — Architecture B.

Every experiment below is PREREGISTERED in code (hypothesis, null,
preregistered metric, baseline, conditions, seed) BEFORE the run. Results
are written to receipts/ as JSON with interpretation and limitations
appended after the run. No retrospective metric selection.

Conventions:
- Closed-loop runs use the canonical flesh-pits environments (read-only;
  imported via sys.path, never modified) under contract v1.0.0.
- One env instance reused across episodes (phase dynamics advance); one
  agent instance reused across episodes (cross-episode learning).
- Per-episode seeds derive from (primary_seed, run_index) via
  env_interface.derive_seed; agent seeds use the "agent" stream.
- Deterministic: no random/time globals; all RNG from seeded Random.

Usage:
    python3 experiments.py --exp K1    # single experiment
    python3 experiments.py --exp ALL   # full battery (K1-K5 + claims)
"""

import argparse
import copy
import datetime
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP_EXP = os.path.join(HERE, "..", "..", "..", "flesh-pits", "experiments")
sys.path.insert(0, FP_EXP)

from env_interface import (  # noqa: E402
    CONTRACT_VERSION, config_hash, derive_seed, new_rng, obs_to_vector,
)
from envs import ALL_ENVS  # noqa: E402

from agent import ArchB  # noqa: E402

ENVS_BY_NAME = {c.NAME: c for c in ALL_ENVS}
RECEIPTS_DIR = os.path.join(HERE, "receipts")
OUT_DIR = os.path.join(HERE, "experiments_out")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def make_agent(env_cls, seed=0, log_path=None, **kwargs):
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    return ArchB(observation_space=space, n_actions=n_actions,
                 env_name=env_cls.NAME, seed=seed,
                 log_path=log_path, **kwargs)


def run_closed_loop(env_name, agent, n_episodes, primary_seed,
                    max_steps=None, record_actions=False):
    """Canonical harness loop. Returns per-episode dicts."""
    env_cls = ENVS_BY_NAME[env_name]
    env = env_cls()
    episodes = []
    for run_index in range(n_episodes):
        episode_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(episode_seed)
        init_hash = env.state_hash()
        agent.reset(derive_seed(episode_seed, run_index, "agent"),
                    env.action_space())
        if record_actions:
            agent._ep_actions = []
        total, steps = 0.0, 0
        e0_hist = []
        done = False
        while not done:
            if max_steps is not None and steps >= max_steps:
                break
            action = agent.act(obs)
            if record_actions:
                agent._ep_actions.append(action)
            obs2, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done, info)
            total += reward
            steps += 1
            obs = obs2
        # Per-tick |e0| for this episode (agent.agg is cumulative; slice it).
        episodes.append({
            "episode": run_index,
            "seed": episode_seed,
            "init_hash": init_hash,
            "final_hash": env.state_hash(),
            "return": total,
            "steps": steps,
            "done": done,
            "actions": list(agent._ep_actions) if record_actions else None,
        })
    return episodes


def collect_transitions(env_name, n_episodes, primary_seed, agent):
    """Open-loop transition collection: (obs_vec, action, obs_next, reward).

    The agent acts closed-loop; transitions are recorded for replay/shuffle
    probes. obs vectors use the env's observation space field order.
    """
    env_cls = ENVS_BY_NAME[env_name]
    env = env_cls()
    space = env.observation_space()
    transitions = []
    for run_index in range(n_episodes):
        episode_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(episode_seed)
        agent.reset(derive_seed(episode_seed, run_index, "agent"),
                    env.action_space())
        done = False
        while not done:
            action = agent.act(obs)
            obs2, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done, info)
            transitions.append((
                obs_to_vector(obs, space), action,
                obs_to_vector(obs2, space), float(reward)))
            obs = obs2
    return transitions


def write_receipt(exp_id, prereg, result, interpretation, limitations):
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {
        "experiment_id": exp_id,
        "architecture": "B",
        "contract_version": CONTRACT_VERSION,
        "preregistered": prereg,
        "config_hash": config_hash(prereg.get("conditions", {})),
        "result": result,
        "interpretation": interpretation,
        "limitations": limitations,
        "written_utc": now,
    }
    path = os.path.join(RECEIPTS_DIR, f"{exp_id}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# K1 — learning freeze
# ---------------------------------------------------------------------------
K1_PREREG = {
    "hypothesis": "On changing_rule, zeroing all update rates degrades "
                  "post-flip performance vs the learning version.",
    "null": "No performance difference (the 'model' is a static function).",
    "metric": "mean_return over episodes 5-9 (post-first-flip), 10 episodes.",
    "baseline": "arch_b frozen (all eta = 0), same action selection.",
    "conditions": {"env": "changing_rule v1.0.0", "episodes": 10,
                   "primary_seed": 101, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
    "kill": "No degradation -> REJECT the architecture as a world model.",
}

def exp_k1():
    prereg = copy.deepcopy(K1_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    learn = make_agent(ENVS_BY_NAME["changing_rule"], seed=11, affect="none")
    eps_learn = run_closed_loop("changing_rule", learn, 10, seed)
    frozen = make_agent(ENVS_BY_NAME["changing_rule"], seed=11, frozen=True,
                        affect="none")
    eps_frozen = run_closed_loop("changing_rule", frozen, 10, seed)
    post_learn = _mean([e["return"] for e in eps_learn[5:10]])
    post_frozen = _mean([e["return"] for e in eps_frozen[5:10]])
    result = {"mean_return_postflip_learn": post_learn,
              "mean_return_postflip_frozen": post_frozen,
              "delta": post_learn - post_frozen,
              "episodes_learn": [e["return"] for e in eps_learn],
              "episodes_frozen": [e["return"] for e in eps_frozen]}
    kill = not (post_learn > post_frozen + 1e-9)
    interp = ("KILL TRIGGERED — no degradation under learning freeze; the "
              "model is a static function." if kill else
              f"Survives: learning improves post-flip return by "
              f"{result['delta']:.3f} (learn {post_learn:.2f} vs frozen "
              f"{post_frozen:.2f}).")
    lim = ("Frozen agent still does mu0 state tracking and active-inference "
           "selection with frozen weights; only parameter learning is zeroed. "
           "Single seed; 10 episodes.")
    path = write_receipt("EXP-AB-K1", prereg, result, interp, lim)
    print(f"K1: learn={post_learn:.3f} frozen={post_frozen:.3f} "
          f"delta={result['delta']:+.3f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# K2 — learned vs fixed predictor (identical action streams)
# ---------------------------------------------------------------------------
K2_PREREG = {
    "hypothesis": "A learned predictor has lower prediction error than a "
                  "fixed (random-init, frozen) predictor on the IDENTICAL "
                  "action stream.",
    "null": "No prediction-error gap (learning machinery is decorative).",
    "metric": "mean |e0| over all ticks of 10 changing_rule episodes, "
              "identical actions replayed.",
    "baseline": "arch_b frozen replaying the learned agent's actions.",
    "conditions": {"env": "changing_rule v1.0.0", "episodes": 10,
                   "primary_seed": 102, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
    "kill": "No gap -> ship the fixed predictor; learning is decorative.",
}

def run_replay(env_name, agent, n_episodes, primary_seed, action_seqs):
    """Replay prescribed action sequences with ONE env instance so the
    episode-indexed phase schedule is identical to the original run."""
    env_cls = ENVS_BY_NAME[env_name]
    env = env_cls()
    for run_index in range(n_episodes):
        episode_seed = derive_seed(primary_seed, run_index, "env")
        obs = env.reset(episode_seed)
        agent.reset(derive_seed(episode_seed, run_index, "agent"),
                    env.action_space())
        agent.replay_actions = list(action_seqs[run_index])
        done = False
        while not done:
            action = agent.act(obs)
            obs2, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done, info)
            obs = obs2


def exp_k2():
    prereg = copy.deepcopy(K2_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    learn = make_agent(ENVS_BY_NAME["changing_rule"], seed=21, affect="none")
    eps = run_closed_loop("changing_rule", learn, 10, seed,
                          record_actions=True)
    err_learn = list(learn.agg["e0_abs"])
    frozen = make_agent(ENVS_BY_NAME["changing_rule"], seed=21, frozen=True,
                        affect="none")
    run_replay("changing_rule", frozen, 10, seed,
               [ep["actions"] for ep in eps])
    err_frozen = list(frozen.agg["e0_abs"])
    m_learn, m_frozen = _mean(err_learn), _mean(err_frozen)
    result = {"mean_abs_e0_learn": m_learn, "mean_abs_e0_frozen": m_frozen,
              "delta": m_frozen - m_learn, "n_ticks": len(err_learn)}
    kill = not (m_learn < m_frozen - 1e-9)
    interp = ("KILL TRIGGERED — learned predictor no better than fixed."
              if kill else
              f"Survives: learned mean|e0|={m_learn:.4f} < fixed "
              f"{m_frozen:.4f} (gap {result['delta']:.4f}).")
    lim = ("One env instance reused across replay episodes so the phase "
           "schedule matches the learned run exactly; cue order and noise "
           "draws are action-independent in the env, so observation streams "
           "are identical and only rewards differ. Prediction quality only "
           "— control is factored out by design.")
    path = write_receipt("EXP-AB-K2", prereg, result, interp, lim)
    print(f"K2: learn_err={m_learn:.4f} frozen_err={m_frozen:.4f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# K3 — hierarchy lesion (structured vs white noise)
# ---------------------------------------------------------------------------
K3_PREREG = {
    "hypothesis": "Disabling L1 raises L0 prediction errors on structured "
                  "tasks (changing_rule) but not on white noise.",
    "null": "Identical performance with/without L1 (hierarchy adds nothing).",
    "metric": "mean |e0| on held-out ordered transitions after training on "
              "300 transitions; structured vs white-noise streams.",
    "baseline": "arch_b with lesion_l1=True (L0-only).",
    "conditions": {"env": "changing_rule v1.0.0", "train_transitions": 300,
                   "primary_seed": 103, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
    "kill": "Identical on structured -> collapse to one level.",
}

def _white_noise_transitions(n, obs_dim, n_actions, seed):
    rng = new_rng(seed)
    out = []
    for _ in range(n):
        o = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        a = rng.randrange(n_actions)
        o2 = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        r = 1.0 if rng.random() < 0.5 else 0.0
        out.append((o, a, o2, r))
    return out

def _train_eval(transitions, heldout, lesion_l1, seed, env_name,
                obs_dim, n_actions):
    space = ENVS_BY_NAME[env_name]().observation_space()
    agent = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, lesion_l1=lesion_l1, seed=seed)
    for (o, a, o2, r) in transitions:
        agent.learn_transition(o, a, o2, r)
    errs = [agent.eval_transition(o, a, o2) for (o, a, o2, r) in heldout]
    return _mean(errs)

def exp_k3():
    prereg = copy.deepcopy(K3_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    env_name = "changing_rule"
    env_cls = ENVS_BY_NAME[env_name]
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    obs_dim = sum(1 if s["type"] == "scalar" else
                  s["shape"] if s["type"] == "vector" else
                  len(s["values"]) for s in space.values())
    ref = make_agent(env_cls, seed=31)
    all_t = collect_transitions(env_name, 12, seed, ref)
    train, heldout = all_t[:300], all_t[300:420]
    wn_train = _white_noise_transitions(300, obs_dim, n_actions, seed + 1)
    wn_held = _white_noise_transitions(120, obs_dim, n_actions, seed + 2)

    s_intact = _train_eval(train, heldout, False, 32, env_name, obs_dim, n_actions)
    s_les = _train_eval(train, heldout, True, 32, env_name, obs_dim, n_actions)
    w_intact = _train_eval(wn_train, wn_held, False, 33, env_name, obs_dim, n_actions)
    w_les = _train_eval(wn_train, wn_held, True, 33, env_name, obs_dim, n_actions)
    result = {"structured_intact": s_intact, "structured_lesioned": s_les,
              "structured_delta": s_les - s_intact,
              "whitenoise_intact": w_intact, "whitenoise_lesioned": w_les,
              "whitenoise_delta": w_les - w_intact}
    kill = not (s_les > s_intact + 1e-9)
    flat_wn = abs(w_les - w_intact) < 0.05 * max(w_intact, 1e-9)
    interp = (f"{'KILL TRIGGERED' if kill else 'Survives'}: structured "
              f"lesion delta={result['structured_delta']:+.4f} "
              f"(intact {s_intact:.4f} vs lesioned {s_les:.4f}); white-noise "
              f"delta={result['whitenoise_delta']:+.4f} "
              f"(flat as predicted: {flat_wn}).")
    lim = ("Open-loop transition training (no retrieval correction/affect); "
           "isolates the weight/precision machinery by design. Single seed.")
    path = write_receipt("EXP-AB-K3", prereg, result, interp, lim)
    print(f"K3: struct d={result['structured_delta']:+.4f} "
          f"wn d={result['whitenoise_delta']:+.4f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# K4 — error-derived affect vs PAD dictionary (competing controllers)
# ---------------------------------------------------------------------------
K4_PREREG = {
    "hypothesis": "The error-derived affect controller achieves higher mean "
                  "return than the PAD-dictionary controller on "
                  "resource_world (foraging + death avoidance).",
    "null": "PAD wins or ties (error-dynamic affect is unnecessary).",
    "metric": "mean_return over 15 episodes.",
    "baseline": "arch_b with affect='pad' (donor-exact PAD mapping).",
    "conditions": {"env": "resource_world v1.0.0", "episodes": 15,
                   "primary_seed": 104, "agent": "arch_b v1"},
    "kill": "PAD >= error-derived -> keep PAD, drop the story.",
}

def exp_k4():
    prereg = copy.deepcopy(K4_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    a_err = make_agent(ENVS_BY_NAME["resource_world"], seed=41, affect="error")
    eps_err = run_closed_loop("resource_world", a_err, 15, seed)
    a_pad = make_agent(ENVS_BY_NAME["resource_world"], seed=41, affect="pad")
    eps_pad = run_closed_loop("resource_world", a_pad, 15, seed)
    m_err = _mean([e["return"] for e in eps_err])
    m_pad = _mean([e["return"] for e in eps_pad])
    result = {"mean_return_error_affect": m_err,
              "mean_return_pad": m_pad, "delta": m_err - m_pad,
              "episodes_error": [e["return"] for e in eps_err],
              "episodes_pad": [e["return"] for e in eps_pad]}
    kill = not (m_err > m_pad + 1e-9)
    interp = ("KILL TRIGGERED — PAD wins/ties; error-derived affect story "
              "dropped, PAD kept as cheaper baseline." if kill else
              f"Survives: error-derived {m_err:.3f} > PAD {m_pad:.3f} "
              f"(delta {result['delta']:+.3f}).")
    lim = ("Both controllers drive identical knobs (explore/learn/persist); "
           "only the affect source differs. resource_world is foraging + "
           "death-avoidance. Single seed, 15 episodes.")
    path = write_receipt("EXP-AB-K4", prereg, result, interp, lim)
    print(f"K4: err_aff={m_err:.3f} pad={m_pad:.3f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# K5 — shuffle test
# ---------------------------------------------------------------------------
K5_PREREG = {
    "hypothesis": "A predictive model trained on temporally shuffled "
                  "transitions predicts a held-out ORDERED stream worse than "
                  "one trained on the ordered stream (it learned the "
                  "temporally-tracked rule, not bag-of-features marginals).",
    "null": "Identical performance (bag-of-features, not temporal).",
    "metric": "mean |e0| on held-out phase-1-rule transitions.",
    "baseline": "arch_b trained on shuffled transition order.",
    "conditions": {"env": "changing_rule v1.0.0",
                   "train": "episodes 0-9 (phases 0,1) seed 105",
                   "heldout": "episodes 5-9 (phase 1 rule) seed 205",
                   "primary_seed": 105, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
    "kill": "Identical -> demote the temporal-prediction claim.",
}

def exp_k5():
    prereg = copy.deepcopy(K5_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    env_name = "changing_rule"
    env_cls = ENVS_BY_NAME[env_name]
    space = env_cls().observation_space()
    n_actions = env_cls().action_space()["n"]
    ref = make_agent(env_cls, seed=51, affect="none")
    train = collect_transitions(env_name, 10, seed, ref)
    ref2 = make_agent(env_cls, seed=52, affect="none")
    heldout = collect_transitions(env_name, 10, seed + 100, ref2)[200:]
    # heldout = episodes 5-9 of the second run = phase 1 rule [1,0],
    # matching the rule at the END of the ordered training stream.
    rng = new_rng(seed + 5)
    shuffled = list(train)
    rng.shuffle(shuffled)
    a_ord = ArchB(observation_space=space, n_actions=n_actions,
                  env_name=env_name, seed=53)
    for (o, a, o2, r) in train:
        a_ord.learn_transition(o, a, o2, r)
    a_shuf = ArchB(observation_space=space, n_actions=n_actions,
                   env_name=env_name, seed=53)
    for (o, a, o2, r) in shuffled:
        a_shuf.learn_transition(o, a, o2, r)
    e_ord = _mean([a_ord.eval_transition(o, a, o2) for (o, a, o2, r) in heldout])
    e_shuf = _mean([a_shuf.eval_transition(o, a, o2) for (o, a, o2, r) in heldout])
    result = {"mean_abs_e0_ordered_train": e_ord,
              "mean_abs_e0_shuffled_train": e_shuf,
              "delta": e_shuf - e_ord, "n_train": len(train),
              "n_heldout": len(heldout)}
    kill = not (e_shuf > e_ord + 1e-9)
    interp = ("KILL TRIGGERED — shuffle makes no difference; the model "
              "learned bag-of-features statistics." if kill else
              f"Survives: shuffled-trained error {e_shuf:.4f} > "
              f"ordered-trained {e_ord:.4f} (delta {result['delta']:+.4f}).")
    lim = ("learn_transition path (no retrieval correction/affect); the "
           "held-out stream uses the phase-1 rule (same as the end of the "
           "ordered training stream) from an independent seed. The "
           "ordered-trained model must track the rule flip to win; the "
           "shuffled-trained model sees mixed rules. Single seed pair.")
    path = write_receipt("EXP-AB-K5", prereg, result, interp, lim)
    print(f"K5: ordered={e_ord:.4f} shuffled={e_shuf:.4f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# Claim 1 — prediction error declines with experience (stationary task)
# ---------------------------------------------------------------------------
C1_PREREG = {
    "hypothesis": "Mean |e0| declines from the first to the last quintile of "
                  "a run on stationary tasks (delayed_reward, grid_world).",
    "null": "No decline (nothing is being learned).",
    "metric": "mean |e0| ticks[0:20%] vs ticks[80:100%].",
    "baseline": "first-quintile self (within-run).",
    "conditions": {"envs": ["delayed_reward v1.0.0", "grid_world v1.0.0"],
                   "episodes": 10, "primary_seed": 106, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
}

def exp_c1():
    prereg = copy.deepcopy(C1_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    out = {}
    for env_name in ("delayed_reward", "grid_world"):
        agent = make_agent(ENVS_BY_NAME[env_name], seed=61, affect="none")
        run_closed_loop(env_name, agent, 10, seed)
        errs = agent.agg["e0_abs"]
        n = len(errs)
        q = max(1, n // 5)
        early, late = _mean(errs[:q]), _mean(errs[-q:])
        out[env_name] = {"early": early, "late": late,
                         "decline": early - late, "n_ticks": n}
    ok = all(v["decline"] > 0 for v in out.values())
    interp = ("Holds" if ok else "FAILS") + ": " + "; ".join(
        f"{k} {v['early']:.4f}->{v['late']:.4f}" for k, v in out.items())
    lim = "Within-run comparison; single seed per env; 10 episodes."
    path = write_receipt("EXP-AB-C1", prereg, out, interp, lim)
    print(f"C1: {interp}\n  {path}")
    return out, not ok


# ---------------------------------------------------------------------------
# Claim 2 — precision weighting beats uniform weighting (noisy task)
# ---------------------------------------------------------------------------
C2_PREREG = {
    "hypothesis": "Estimated precision (pi from error statistics) beats "
                  "uniform pi=1 on the noisy changing_rule task (10% noise).",
    "null": "No decision-quality gap (precision is numerology).",
    "metric": "mean_return over 10 episodes.",
    "baseline": "arch_b with uniform_precision=True (all pi = 1).",
    "conditions": {"env": "changing_rule v1.0.0", "episodes": 10,
                   "primary_seed": 107, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
}

def exp_c2():
    prereg = copy.deepcopy(C2_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    a_prec = make_agent(ENVS_BY_NAME["changing_rule"], seed=71, affect="none")
    eps_prec = run_closed_loop("changing_rule", a_prec, 10, seed)
    a_uni = make_agent(ENVS_BY_NAME["changing_rule"], seed=71,
                       uniform_precision=True, affect="none")
    eps_uni = run_closed_loop("changing_rule", a_uni, 10, seed)
    m_prec = _mean([e["return"] for e in eps_prec])
    m_uni = _mean([e["return"] for e in eps_uni])
    result = {"mean_return_precision": m_prec, "mean_return_uniform": m_uni,
              "delta": m_prec - m_uni}
    kill = not (m_prec > m_uni + 1e-9)
    interp = ("KILL — precision adds nothing; treat as numerology." if kill
              else f"Holds: precision {m_prec:.3f} > uniform {m_uni:.3f}.")
    lim = "Single seed; 10 episodes; 10% label noise in env."
    path = write_receipt("EXP-AB-C2", prereg, result, interp, lim)
    print(f"C2: prec={m_prec:.3f} uniform={m_uni:.3f} kill={kill}\n  {path}")
    return result, kill


# ---------------------------------------------------------------------------
# Claim 4 — active inference beats random and greedy on information gain
# ---------------------------------------------------------------------------
C4_PREREG = {
    "hypothesis": "Active-inference action selection beats random and greedy "
                  "baselines on information-gain metrics (same world model, "
                  "only the action policy differs).",
    "null": "No IG gap ('active inference' is a label on a heuristic).",
    "metric": "IG = mean|e0| first 15% ticks minus mean|e0| last 15% ticks; "
              "secondary: mean_return. 15 episodes on pomaze.",
    "baseline": "arch_b with action_mode random / greedy.",
    "conditions": {"env": "pomaze v1.0.0", "episodes": 15,
                   "primary_seed": 108, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
}

def exp_c4():
    prereg = copy.deepcopy(C4_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    out = {}
    for mode in ("active_inference", "random", "greedy"):
        agent = make_agent(ENVS_BY_NAME["pomaze"], seed=81,
                           action_mode=mode, affect="none")
        run_closed_loop("pomaze", agent, 15, seed)
        errs = agent.agg["e0_abs"]
        n = len(errs)
        q = max(1, int(n * 0.15))
        ig = _mean(errs[:q]) - _mean(errs[-q:])
        # mean return from runner episodes
        out[mode] = {"IG": ig, "n_ticks": n}
    # rerun for returns (cheap, deterministic)
    for mode in out:
        agent = make_agent(ENVS_BY_NAME["pomaze"], seed=81,
                           action_mode=mode, affect="none")
        eps = run_closed_loop("pomaze", agent, 15, seed)
        out[mode]["mean_return"] = _mean([e["return"] for e in eps])
    ig_ai = out["active_inference"]["IG"]
    kill = not (ig_ai > out["random"]["IG"] + 1e-9
                and ig_ai > out["greedy"]["IG"] + 1e-9)
    interp = (f"{'KILL — label comes off' if kill else 'Holds'}: IG "
              f"ai={ig_ai:.4f} rand={out['random']['IG']:.4f} "
              f"greedy={out['greedy']['IG']:.4f}; returns "
              f"{ {m: round(out[m]['mean_return'], 3) for m in out} }.")
    lim = ("Same world model in all conditions; only act() policy differs. "
           "IG measured as within-run error decline. Single seed.")
    path = write_receipt("EXP-AB-C4", prereg, out, interp, lim)
    print(f"C4: {interp}\n  {path}")
    return out, kill


# ---------------------------------------------------------------------------
# Concrete bar — intact vs L1-lesioned on changing_rule (+ beat D's 20/40)
# ---------------------------------------------------------------------------
BAR_PREREG = {
    "hypothesis": "Intact Architecture B shows prediction-error decline on "
                  "changing_rule and beats chance (20/40); the L1-lesioned "
                  "version shows no error decline.",
    "null": "No error decline intact, or lesioned matches intact.",
    "metric": "mean|e0| first 20% vs last 20% of ticks; reward-channel "
              "|e0| decline (the contingency lives in the last_reward "
              "channel); mean_return vs 20.0.",
    "baseline": "arch_b lesion_l1=True; Architecture D chance level 20/40.",
    "conditions": {"env": "changing_rule v1.0.0", "episodes": 10,
                   "primary_seed": 109, "agent": "arch_b v1",
                   "affect": "none (world-model isolation; affect in K4)"},
}

def exp_bar():
    prereg = copy.deepcopy(BAR_PREREG)
    seed = prereg["conditions"]["primary_seed"]
    logp = os.path.join(OUT_DIR, "EXP-AB-BAR.predictions.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    intact = make_agent(ENVS_BY_NAME["changing_rule"], seed=91,
                        log_path=logp, affect="none")
    eps_i = run_closed_loop("changing_rule", intact, 10, seed)
    if intact.plog:
        intact.plog.close()
    les = make_agent(ENVS_BY_NAME["changing_rule"], seed=91, lesion_l1=True,
                       affect="none")
    eps_l = run_closed_loop("changing_rule", les, 10, seed)

    def decline(errs):
        n = len(errs)
        q = max(1, n // 5)
        return _mean(errs[:q]) - _mean(errs[-q:])

    d_i, d_l = decline(intact.agg["e0_abs"]), decline(les.agg["e0_abs"])
    dr_i = decline(intact.agg["e0_reward"])
    dr_l = decline(les.agg["e0_reward"])
    r_i = _mean([e["return"] for e in eps_i])
    r_l = _mean([e["return"] for e in eps_l])
    result = {"intact_error_decline": d_i, "lesioned_error_decline": d_l,
              "intact_rewardchan_decline": dr_i,
              "lesioned_rewardchan_decline": dr_l,
              "intact_mean_return": r_i, "lesioned_mean_return": r_l,
              "chance_level": 20.0,
              "intact_beats_chance": r_i > 20.0,
              "prediction_log": logp,
              "calibration": intact.plog.calibration_summary()
              if intact.plog else None}
    fail = not (dr_i > 1e-9 and dr_l <= dr_i and r_i > 20.0)
    interp = (f"{'BAR NOT CLEARED' if fail else 'BAR CLEARED'}: intact "
              f"decline={d_i:+.4f} (reward-chan {dr_i:+.4f}), lesioned "
              f"decline={d_l:+.4f} (reward-chan {dr_l:+.4f}), intact "
              f"return={r_i:.2f} vs chance 20.0, lesioned return={r_l:.2f}.")
    lim = ("Single seed; 10 episodes; per-tick §30 records in "
           "experiments_out/EXP-AB-BAR.predictions.jsonl.")
    path = write_receipt("EXP-AB-BAR", prereg, result, interp, lim)
    print(f"BAR: {interp}\n  {path}")
    return result, fail


EXPS = {
    "K1": exp_k1, "K2": exp_k2, "K3": exp_k3, "K4": exp_k4, "K5": exp_k5,
    "C1": exp_c1, "C2": exp_c2, "C4": exp_c4, "BAR": exp_bar,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL",
                    choices=sorted(EXPS) + ["ALL"])
    args = ap.parse_args(argv)
    names = sorted(EXPS) if args.exp == "ALL" else [args.exp]
    summary = {}
    for name in names:
        print(f"=== EXP-AB-{name} ===")
        try:
            result, bad = EXPS[name]()
            summary[name] = {"kill_or_fail": bool(bad)}
        except Exception as ex:  # fail visible, never silent
            print(f"EXP-AB-{name} ERROR: {type(ex).__name__}: {ex}")
            summary[name] = {"error": f"{type(ex).__name__}: {ex}"}
    print("\n=== SUMMARY ===")
    for name, s in summary.items():
        print(f"EXP-AB-{name}: {s}")
    with open(os.path.join(OUT_DIR, "battery_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
