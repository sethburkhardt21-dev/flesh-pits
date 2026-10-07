"""Architecture C — hybrid agent (EXP-FP-C-BUILD-AND-BEAT).

Assembly (frozen in experiments/preregistration_ARCH_C.json):
  - A's WorkspaceTick skeleton (imported UNCHANGED): bounded buffer,
    recurrent ignition gate, sole-path broadcast with the 6 declared
    consumers, R1 sub-ignition exploratory path as the default action
    selection, learned-gain attention loop (+ cue-indexed adapter where
    the env exposes a cue) closing through the tick's feedback broadcast.
  - B's HierarchicalGenerativeModel (imported UNCHANGED): the learned
    predictor core. It is the ONLY stimulus source for C's specialists:
    stimulus_a = rhat_a + kappa * mem_bonus_a.
  - B's EpisodicStore (imported UNCHANGED): encodes (obs, action, reward,
    e0, e1, pi, attention=surprise) with provenance; feeds the per-action
    predicted-error bonus via predicted_error_for_action.

SEAM REIMPLEMENTATION (this file only — stated in the preregistration):
  1. predictor-driven specialist factory (the integration surface);
  2. categorical-slice / context-key helpers — an attributed copy of
     ArchB's 8-line helpers (agent.py::_categorical_slices /
     HierarchicalGenerativeModel::context_key), needed because the
     predictor's ctx must be computed from the observation dict;
  3. CAdapter: presents A's tick protocol (observe() / step(label)->reward)
     while exposing the driver protocol (last_transition / reset_episode)
     the learning close needs;
  4. the per-tick learning close (model.observe + memory.store) — the same
     close ArchB performs, minus affect and §30 logging (neither is part
     of C's assembly).

Excluded with reasons (preregistered): PAD (no genuine consumer in C),
error-affect (rejected), ActiveInferenceSelector (IG label killed),
UsefulnessPredictor/RetrievalGate (uncalibrated + degenerate policy),
precision variants (rejected/shelved), consolidation (no online surface).

Stdlib only. Deterministic given seeds. No identity machinery.
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_A = os.path.join(os.path.dirname(_HERE), "architecture-a")
_ARCH_B = os.path.join(os.path.dirname(_HERE), "architecture-b")
_FLESH_EXP = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                          "experiments")
for _p in (_ARCH_A, _ARCH_B, _FLESH_EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- A components: imported unchanged ---------------------------------------
from tick import WorkspaceTick  # noqa: E402
from attention_cue import CueIndexedArbitrator  # noqa: E402
from attention import AttentionArbitrator  # noqa: E402

# --- B components: imported unchanged --------------------------------------
from generative_model import HierarchicalGenerativeModel  # noqa: E402
from memory import EpisodicStore, DisabledStore  # noqa: E402

# --- env contract -----------------------------------------------------------
from env_interface import obs_to_vector, derive_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Seam 2: context helpers (attributed copy of ArchB's helpers).
# ArchB computes these from (observation_space, obs_vec); C needs the same
# mapping so the predictor's ctx matches B's semantics exactly.
# ---------------------------------------------------------------------------
def categorical_slices(observation_space: dict):
    """(start, end) slices of the flattened obs vector for each categorical
    field. Attributed copy of ArchB._categorical_slices."""
    slices = []
    idx = 0
    for spec in observation_space.values():
        t = spec["type"]
        size = (1 if t == "scalar" else spec["shape"]
                if t == "vector" else len(spec["values"]))
        if t == "categorical":
            slices.append((idx, idx + size))
        idx += size
    return slices


def context_key(obs_vec, cat_slices):
    """Discrete context = argmax of each categorical slice. Attributed copy
    of HierarchicalGenerativeModel.context_key."""
    key = []
    for (s, e) in cat_slices:
        seg = obs_vec[s:e]
        key.append(max(range(len(seg)), key=lambda i: seg[i]))
    return tuple(key)


# ---------------------------------------------------------------------------
# Seam 1: predictor-driven specialists — the integration surface.
# ---------------------------------------------------------------------------
def make_predictor_specialists(channels, predictor, memory, observation_space,
                               cat_slices, kappa=0.1):
    """One specialist per action channel. Stimulus = the predictor's
    learned reward expectation for that action in the current context,
    plus a memory-derived exploration bonus (predicted per-action error).

    The predictor LEARNS online (model.observe in the driver's close), so
    the stimuli move with experience — this is the B->A causal surface.
    The memory bonus is None-guarded (honest, not invented) exactly as in
    ArchB's ehat path.

    Non-finite stimuli fail closed (raised, not clamped): a diverging
    predictor is a finding, not a number to paper over.
    """
    specs = []
    for a_idx, c in enumerate(channels):
        def fn(obs, tick, _a=a_idx, _c=c):
            vec = obs_to_vector(obs, observation_space)
            ctx = context_key(vec, cat_slices)
            rhat = predictor.predict_reward(list(predictor.mu0), _a, ctx)
            pe = memory.predicted_error_for_action(vec, _a)
            bonus = kappa * pe if pe is not None else 0.0
            stim = float(rhat + bonus)
            if not math.isfinite(stim):
                raise ValueError(
                    f"non-finite C stimulus on channel {_c!r}: "
                    f"rhat={rhat!r} pe={pe!r}")
            return stim, {"summary": f"predictor:{_c}", "rhat": float(rhat),
                          "pe_bonus": float(bonus)}
        specs.append((c, fn))
    return specs


# ---------------------------------------------------------------------------
# Seam 3: CAdapter — A's tick protocol + the driver's transition protocol.
# ---------------------------------------------------------------------------
class CAdapter:
    """Adapts a canonical Environment to Architecture A's tick protocol
    (observe() -> dict for specialists, step(label) -> float reward) while
    exposing the full transition to the driver for the learning close."""

    def __init__(self, env, channel_to_action: dict):
        self.env = env
        self.channel_to_action = dict(channel_to_action)
        self.obs = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        self.episode_idx = -1

    def reset_episode(self, seed: int) -> dict:
        self.obs = self.env.reset(seed)
        self.episode_idx += 1
        self.last_done = False
        return dict(self.obs)

    def observe(self) -> dict:
        return dict(self.obs)

    def step(self, action_label) -> float:
        """Tick protocol: action label in -> float reward out."""
        a = self.channel_to_action[action_label]
        obs, reward, done, info = self.env.step(a)
        self.obs = obs
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = dict(info)
        return float(reward)

    def last_transition(self):
        """Driver protocol: (obs_next, reward, done, info) of the last step."""
        return (dict(self.obs), self.last_reward, self.last_done,
                dict(self.last_info))


# ---------------------------------------------------------------------------
# Seam 4: the per-tick learning close (ArchB's close, minus affect/§30).
# ---------------------------------------------------------------------------
def close_tick_c(predictor, memory, *, state_vec, action_idx, ctx, obs_vec,
                 obs_next_vec, reward, tick, episode_idx, env_name) -> dict:
    """Close one C tick: predictor belief update + episodic encoding.

    Mirrors ArchB._close_tick's learning path exactly (same observe call,
    same surprise-derived attention formula, same store signature).
    Returns diagnostics (not used for decisions).
    """
    diag = predictor.observe(state_vec, action_idx, ctx, obs_next_vec,
                            reward)
    e0_abs = math.sqrt(sum(e * e for e in diag["e0"]))
    _oconf, ounc = predictor.obs_confidence()
    attention = max(0.0, min(1.0, e0_abs / (1.0 + ounc)))
    mem_id = memory.store(obs_vec, action_idx, reward, diag["e0"],
                          diag["e1"], diag["pi0"], attention, tick,
                          episode_idx, env_name)
    return {"e0_abs": e0_abs, "ounc": ounc, "attention": attention,
            "mem_id": mem_id, "update_norm": diag["update_norm"],
            "n_contexts": diag["n_contexts"]}


# ---------------------------------------------------------------------------
# C agent assembly.
# ---------------------------------------------------------------------------
class ArchC:
    """Architecture C: A's workspace tick driven by B's predictor + memory.

    config keys (all frozen by the preregistration):
      channels, channel_to_action, observation_space, env_name,
      context_fn (or None), gain_lr, theta, reward_baseline (fixed 0.5),
      kappa, predictor_kwargs, memory_enabled, agent_seed.
    """

    NAME = "arch_c"

    def __init__(self, config: dict):
        self.config = dict(config)
        channels = list(config["channels"])
        self.channels = channels
        self.channel_to_action = dict(config["channel_to_action"])
        self.observation_space = dict(config["observation_space"])
        self.env_name = config["env_name"]
        self.cat_slices = categorical_slices(self.observation_space)
        self.kappa = float(config.get("kappa", 0.1))
        seed = int(config.get("agent_seed", 0))

        # B components (imported unchanged).
        pk = dict(config.get("predictor_kwargs", {}))
        pk.setdefault("seed", seed)
        self.predictor = HierarchicalGenerativeModel(
            obs_dim=sum(1 if s["type"] == "scalar" else s["shape"]
                        if s["type"] == "vector" else len(s["values"])
                        for s in self.observation_space.values()),
            n_actions=len(channels), cat_slices=self.cat_slices, **pk)
        self.memory = (EpisodicStore() if config.get("memory_enabled", True)
                       else DisabledStore())
        self.frozen_predictor = bool(config.get("frozen_predictor", False))
        if self.frozen_predictor:
            self.predictor.eta0 = 0.0
            self.predictor.etaD = 0.0
            self.predictor.eta_r = 0.0
            self.predictor.etaR = 0.0

        # A components (imported unchanged): the tick with
        # predictor-driven specialists. R1 default action selection is
        # inside WorkspaceTick._select_action (K8-wired).
        specialists = make_predictor_specialists(
            channels, self.predictor, self.memory, self.observation_space,
            self.cat_slices, kappa=self.kappa)
        context_fn = config.get("context_fn")
        arb_cls = (CueIndexedArbitrator if context_fn is not None
                   else AttentionArbitrator)
        self.tick = WorkspaceTick(
            channels, specialists, capacity=len(channels),
            gain_lr=float(config.get("gain_lr", 0.15)),
            frozen_gains=bool(config.get("frozen_gains", False)),
            ignition_kwargs={"theta": float(config.get("theta", 0.45))},
            arbitrator_cls=arb_cls, context_fn=context_fn)

        self._tick_index = 0
        self._episode_idx = -1
        self._state_vec = None  # belief used for the current tick's stimuli

    # -- episode handling ---------------------------------------------------
    def reset_episode(self, obs: dict) -> None:
        """Episode boundary: seed the predictor belief from the first
        observation (B's convention), exactly as ArchB.act does."""
        self._episode_idx += 1
        vec = obs_to_vector(obs, self.observation_space)
        self.predictor.mu0 = list(vec)
        self._state_vec = list(vec)

    # -- one closed-loop tick ------------------------------------------------
    def step(self, adapter: CAdapter) -> dict:
        """One C tick: A's tick (stimuli from the predictor) + the B
        learning close. Returns the tick trace plus C diagnostics."""
        self._tick_index += 1
        obs = adapter.observe()
        obs_vec = obs_to_vector(obs, self.observation_space)
        ctx = context_key(obs_vec, self.cat_slices)
        state_vec = list(self.predictor.mu0)
        self._state_vec = state_vec

        trace = self.tick.step(obs, adapter)
        action_label = trace["action"]
        action_idx = self.channel_to_action[action_label]
        reward = trace["reward"]
        obs_next, _r2, _done, _info = adapter.last_transition()
        obs_next_vec = obs_to_vector(obs_next, self.observation_space)

        diag = close_tick_c(
            self.predictor, self.memory, state_vec=state_vec,
            action_idx=action_idx, ctx=ctx, obs_vec=obs_vec,
            obs_next_vec=obs_next_vec, reward=reward,
            tick=self._tick_index, episode_idx=self._episode_idx,
            env_name=self.env_name)
        trace["c_diagnostics"] = diag
        trace["action_idx"] = action_idx
        return trace

    def stats(self) -> dict:
        s = self.tick.stats()
        s.update({"memory_size": len(self.memory),
                  "n_contexts": self.predictor.n_contexts,
                  "tick_index": self._tick_index})
        return s
