"""ArchB — Architecture B agent (predictive-model-centric).

Implements the flesh-pits Agent ABC (contract v1.0.0):

  reset(seed, action_space)  per-episode state reset; learned parameters
                             persist across episodes in the agent object.
  act(obs)                   1. closes the previous tick (prediction error,
                                precision, belief updates, memory, affect,
                                prediction-target logging);
                             2. selects an action (active inference / greedy /
                                random / replay);
                             3. issues the three §30 predictions for the tick.
  update(obs, action, reward, done, info)
                             records the transition outcome for the pending
                             prediction (reward/done). info is metadata only
                             and is never used for decisions.
  snapshot()/restore()       JSON-serializable full internal state.

Per-tick data flow::

    obs_t --[predict]--> xhat(a), rhat(a), uhat --[act]--> a_t
        --env--> obs_{t+1}, r_t --[close tick]--> e0, e1, pi, dmu, memory,
                 affect, §30 records --[predict]--> ...

The generative model's belief states mu0/mu1 persist across ticks AND
across episodes (cross-episode learning); only per-episode buffers
(pending transition, tick counter, RNG stream) reset.

Configuration knobs (all honest ablations):
  action_mode: "active_inference" | "greedy" | "random"
  affect:      "error" | "pad" | "none"
  lesion_l1:   True disables the L1 top-down path (K3)
  uniform_precision: True forces all pi = 1 (claim-2 ablation)
  memory_enabled: False swaps in DisabledStore (no encoding/retrieval;
                  M1 no-memory ablation)
  precision_kind: "estimated" | "uniform" | "shift_reset" — shift-robust
                  precision variant (C2B); surprise_k sets the change-point
                  threshold
  frozen:      True zeroes every learning rate (K1/K2)
  replay_actions: optional list[int] — act() consumes these instead of
                  selecting (K2 identical-stream replay)
  gate_policy:    "ungated" (default; current behavior, correction applied
                  whenever available) | "uhat" (apply iff uhat > threshold)
                  | "random" (apply w.p. rate; chance control)
  gate_threshold: uhat decision boundary (default 0.0: positive predicted
                  benefit)
  gate_rate:      application probability for the random gate
  gate_seed:      seed for the random gate's RNG
  gate_uhat_source: "live" (default; the gate reads the live
                  UsefulnessPredictor's uhat) | "shadow" (EXP-FP-0007: the
                  gate reads a second UsefulnessPredictor trained on the
                  counterfactual UNCONDITIONAL benefit, so gating never
                  contaminates its training data; the live predictor keeps
                  training on realized benefit and is not read by the gate)
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "..", "flesh-pits", "experiments"))

from env_interface import Agent, obs_to_vector  # noqa: E402

from generative_model import HierarchicalGenerativeModel  # noqa: E402
from memory import EpisodicStore, DisabledStore  # noqa: E402
from predictions import PredictionLog, UsefulnessPredictor  # noqa: E402
from retrieval_gate import RetrievalGate  # noqa: E402
from active_inference import ActiveInferenceSelector  # noqa: E402
from affect import ErrorAffect, PadController  # noqa: E402


class ArchB(Agent):
    NAME = "arch_b"

    def __init__(self, observation_space: dict, n_actions: int,
                 env_name: str = "unknown",
                 action_mode: str = "active_inference",
                 affect: str = "error",
                 lesion_l1: bool = False,
                 uniform_precision: bool = False,
                 memory_enabled: bool = True,
                 precision_kind: str = "estimated",
                 surprise_k: float = 4.0,
                 frozen: bool = False,
                 lamV: float = 1.0, lamIG: float = 0.5, lamR: float = 0.10,
                 seed: int = 0,
                 log_path: Optional[str] = None,
                 gate_policy: str = "ungated",
                 gate_threshold: float = 0.0,
                 gate_rate: float = 1.0,
                 gate_seed: int = 0,
                 gate_uhat_source: str = "live") -> None:
        self.observation_space = dict(observation_space)
        self.n_actions = n_actions
        self.env_name = env_name
        self.action_mode = action_mode
        self.affect_kind = affect
        self.lesion_l1 = lesion_l1
        self.uniform_precision = uniform_precision
        self.memory_enabled = memory_enabled
        self.precision_kind = precision_kind
        self.surprise_k = surprise_k
        self.frozen = frozen
        self.seed = seed
        self.log_path = log_path

        self.obs_dim = self._obs_dim()
        self._reward_chan_idx = self._find_reward_channel()
        self._cat_slices = self._categorical_slices()
        etas = dict(eta0=0.0, etaD=0.0, eta_r=0.0,
                    etaR=0.0) if frozen else {}
        self.model = HierarchicalGenerativeModel(
            self.obs_dim, n_actions, cat_slices=self._cat_slices,
            uniform_precision=uniform_precision,
            precision_kind=precision_kind, surprise_k=surprise_k,
            seed=seed, **etas)
        self.base_etas = (self.model.eta0, self.model.etaD,
                          self.model.eta_r, self.model.etaR)
        self.memory = EpisodicStore() if memory_enabled else DisabledStore()
        self.selector = ActiveInferenceSelector(
            n_actions, lamV=lamV, lamIG=lamIG, lamR=lamR,
            mode=action_mode if action_mode != "replay" else "random",
            seed=seed + 1)
        self.err_affect = ErrorAffect()
        self.pad = PadController()
        self.usefulness = UsefulnessPredictor()
        self.gate = RetrievalGate(policy=gate_policy,
                                  threshold=gate_threshold,
                                  rate=gate_rate, seed=gate_seed)
        if gate_uhat_source not in ("live", "shadow"):
            raise ValueError(
                f"unknown gate_uhat_source: {gate_uhat_source!r}")
        self.gate_uhat_source = gate_uhat_source
        # EXP-FP-0007 shadow predictor: a second UsefulnessPredictor used
        # ONLY as the gate's uhat source in shadow mode. It trains on the
        # counterfactual unconditional benefit every available-correction
        # tick, so the gate's decisions never contaminate its training
        # data. The live predictor above is untouched (still trains on
        # realized benefit, still feeds the §30 records).
        self.shadow_usefulness = UsefulnessPredictor()
        # Run diagnostic: (uhat_gate, benefit_shadow) per shadow tick.
        self._shadow_hist: List[tuple] = []
        self.plog: Optional[PredictionLog] = None

        self.replay_actions: Optional[List[int]] = None
        self._rng = random.Random(seed + 2)
        self._episode = -1
        self._tick = 0
        self._pending: Optional[Dict] = None
        self._last_action: Optional[int] = None
        self._e1_ema = 0.0
        self.knobs = {"explore_gain": 1.0, "learn_gain": 1.0, "persist": 0.0,
                      "valence": 0.0, "arousal": 0.0}
        # Per-run aggregates for receipts.
        self.agg = {"e0_abs": [], "e1_abs": [], "e0_reward": [],
                    "returns": []}

    def _obs_dim(self) -> int:
        d = 0
        for spec in self.observation_space.values():
            t = spec["type"]
            if t == "scalar":
                d += 1
            elif t == "vector":
                d += spec["shape"]
            elif t == "categorical":
                d += len(spec["values"])
        return d

    def _find_reward_channel(self) -> Optional[int]:
        """Index of the scalar obs channel carrying reward information
        (name contains 'reward'), for reward-channel error tracking.
        Generic: works for any env that surfaces reward in its obs."""
        idx = 0
        for name, spec in self.observation_space.items():
            t = spec["type"]
            size = (1 if t == "scalar" else spec["shape"]
                    if t == "vector" else len(spec["values"]))
            if t == "scalar" and "reward" in name:
                return idx
            idx += size
        return None

    def _categorical_slices(self) -> List[Tuple[int, int]]:
        """(start, end) slices of the flattened obs vector for each
        categorical field — used to extract the discrete context key."""
        slices = []
        idx = 0
        for spec in self.observation_space.values():
            t = spec["type"]
            size = (1 if t == "scalar" else spec["shape"]
                    if t == "vector" else len(spec["values"]))
            if t == "categorical":
                slices.append((idx, idx + size))
            idx += size
        return slices

    def context_key(self, obs_vec: List[float]) -> Tuple[int, ...]:
        return self.model.context_key(obs_vec)

    def _vec(self, obs: dict) -> List[float]:
        return obs_to_vector(obs, self.observation_space)

    # -- Agent ABC ---------------------------------------------------------
    def reset(self, seed: int, action_space: dict) -> None:
        if action_space.get("n") != self.n_actions:
            raise ValueError("action space mismatch")
        self._episode += 1
        self._tick = 0
        self._pending = None
        self._last_action = None
        self._e1_ema = 0.0
        self._rng = random.Random(seed ^ 0x9E3779B9)
        self.selector._rng = random.Random((seed ^ 0x9E3779B9) + 7)
        if self.plog is None and self.log_path:
            self.plog = PredictionLog(self.log_path)

    def act(self, obs: dict) -> int:
        obs_vec = self._vec(obs)
        if self._pending is not None:
            self._close_tick(obs_vec)
        if self._tick == 0:
            # Prior state belief = first observation of the episode.
            self.model.mu0 = list(obs_vec)
        action = self._select_action(obs_vec)
        self._open_tick(obs_vec, action)
        self._last_action = action
        return action

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        # info is metadata only — never used for decisions (contract).
        if self._pending is None:
            raise RuntimeError("update with no pending prediction")
        if action != self._pending["action"]:
            raise RuntimeError("action mismatch between act and update")
        self._pending["reward"] = float(reward)
        self._pending["done"] = bool(done)
        self.agg["returns"][-1] += float(reward)

    # -- tick internals ------------------------------------------------------
    def _select_action(self, obs_vec: List[float]) -> int:
        if self.replay_actions:
            a = self.replay_actions.pop(0)
            if not 0 <= a < self.n_actions:
                raise ValueError("replay action out of range")
            return a
        persist = self.knobs["persist"]
        last = self._last_action
        lamIG_eff = self.selector.lamIG * self.knobs["explore_gain"]
        state_vec = list(self.model.mu0)
        ctx = self.context_key(obs_vec)

        def vhat(a: int) -> float:
            v = self.model.predict_reward(state_vec, a, ctx)
            if last is not None and a == last:
                v += persist
            return v

        pi1_mean = sum(self.model.last_pi1) / max(1, len(self.model.last_pi1))

        def ehat(a: int) -> float:
            mem = self.memory.predicted_error_for_action(obs_vec, a)
            base = mem if mem is not None else self.selector.table_error(a)
            return pi1_mean * base

        risk = math.sqrt(
            sum(self.model.prec0.channel_variance()) /
            max(1, self.obs_dim) + 1e-9)
        # Temporarily scale the IG weight by the affect knob.
        lamIG_saved = self.selector.lamIG
        self.selector.lamIG = lamIG_eff
        try:
            return self.selector.select(vhat, ehat, risk)
        finally:
            self.selector.lamIG = lamIG_saved

    def _open_tick(self, obs_vec: List[float], action: int) -> None:
        state_vec = list(self.model.mu0)
        ctx = self.context_key(obs_vec)
        xhat_raw = self.model.predict_next(state_vec, action, ctx)
        corr = self.memory.retrieval_correction(obs_vec, k=5)
        correction = corr["correction"]
        rhat = self.model.predict_reward(state_vec, action, ctx)
        rconf, runc = self.model.reward_confidence()
        oconf, ounc = self.model.obs_confidence()
        qfeat = self.usefulness.features(
            corr["mean_similarity"], corr["n"], ounc, self._e1_ema)
        uhat = self.usefulness.predict(qfeat)
        uconf, uunc = self.usefulness.confidence()
        # §9 item 6: the gate reads uhat but never modifies the predictor.
        # With policy "ungated" this reduces to the original behavior
        # (decide -> True, correction applied whenever available).
        # EXP-FP-0007: gate_uhat_source="shadow" reads the shadow
        # predictor's uhat instead of the live one.
        uhat_gate = uhat
        if correction is not None and self.gate_uhat_source == "shadow":
            uhat_gate = self.shadow_usefulness.predict(qfeat)
        apply = (self.gate.decide(uhat_gate)
                 if correction is not None else False)
        if correction is not None and apply:
            xhat = [x + c for x, c in zip(xhat_raw, correction)]
            retrieval_used = True
        else:
            xhat = list(xhat_raw)
            retrieval_used = False
        self._pending = {
            "tick": self._tick, "episode": self._episode,
            "obs_vec": obs_vec, "state_vec": state_vec, "action": action,
            "ctx": ctx,
            "xhat_raw": xhat_raw, "xhat": xhat,
            "retrieval_used": retrieval_used,
            "mean_similarity": corr["mean_similarity"],
            "n_nbrs": corr["n"],
            "rhat": rhat, "rconf": rconf, "runc": runc,
            "oconf": oconf, "ounc": ounc,
            "uhat": uhat, "uconf": uconf, "uunc": uunc,
            "uhat_gate": uhat_gate,
            "correction": correction,
            "qfeat": qfeat,
            "reward": 0.0, "done": False,
        }
        self.agg["returns"].append(0.0)

    def _close_tick(self, obs_next: List[float]) -> None:
        p = self._pending
        assert p is not None
        # The agent's prediction error (perception IS prediction error).
        e0 = [o - x for o, x in zip(obs_next, p["xhat"])]
        e0_raw = [o - x for o, x in zip(obs_next, p["xhat_raw"])]
        e0_abs = math.sqrt(sum(e * e for e in e0))
        e0_raw_abs = math.sqrt(sum(e * e for e in e0_raw))

        # Apply affect-scaled learning rates for this tick's update.
        lg = self.knobs["learn_gain"]
        e0s, eDs, ers, eRs = self.base_etas
        self.model.eta0, self.model.etaD = e0s * lg, eDs * lg
        self.model.eta_r, self.model.etaR = ers * lg, eRs * lg
        try:
            diag = self.model.observe(
                p["state_vec"], p["action"], p["ctx"], obs_next,
                p["reward"], lesion_l1=self.lesion_l1)
        finally:
            (self.model.eta0, self.model.etaD,
             self.model.eta_r, self.model.etaR) = self.base_etas
        e1_abs = math.sqrt(sum(e * e for e in diag["e1"]))
        self._e1_ema += 0.1 * (e1_abs - self._e1_ema)
        self.selector.observe_error(p["action"], e1_abs)
        self.agg["e0_abs"].append(e0_abs)
        self.agg["e1_abs"].append(e1_abs)
        if self._reward_chan_idx is not None:
            self.agg["e0_reward"].append(abs(e0[self._reward_chan_idx]))

        # Affect update from error dynamics / reward.
        ounc = p["ounc"]
        if self.affect_kind == "error":
            self.knobs = self.err_affect.update(e0_abs, ounc)
        elif self.affect_kind == "pad":
            self.knobs = self.pad.update(p["reward"], e0_abs, ounc)
        else:
            self.knobs = {"explore_gain": 1.0, "learn_gain": 1.0,
                          "persist": 0.0, "valence": 0.0, "arousal": 0.0}

        # Episodic encoding with provenance; attention = surprise.
        attention = max(0.0, min(1.0, e0_abs / (1.0 + ounc)))
        self.memory.store(
            p["obs_vec"], p["action"], p["reward"], e0, diag["e1"],
            diag["pi0"], attention, self._tick, self._episode, self.env_name)

        # §30 records.
        if self.plog is not None:
            self.plog.record_next_obs(
                self._tick, self._episode, p["xhat"], p["oconf"], p["ounc"],
                obs_next, diag["update_norm"])
            self.plog.record_action_consequence(
                self._tick, self._episode, p["rhat"], p["rconf"], p["runc"],
                p["reward"], diag["update_norm"])
            benefit = e0_raw_abs - e0_abs  # measured retrieval benefit
            upd = self.usefulness.update(p["qfeat"], benefit)
            self.plog.record_retrieval_usefulness(
                self._tick, self._episode, p["uhat"], p["uconf"],
                p["uunc"], benefit, upd["update_norm"])
        else:
            self.usefulness.update(p["qfeat"], e0_raw_abs - e0_abs)

        # EXP-FP-0007 shadow predictor: trains on the counterfactual
        # UNCONDITIONAL benefit (what applying the correction would have
        # done), so gating never contaminates its training data. The live
        # predictor above keeps training on realized benefit, exactly as
        # in EXP-FP-0006.
        if (self.gate_uhat_source == "shadow"
                and p.get("correction") is not None):
            corr_vec = p["correction"]
            xhat_applied = [x + c for x, c in zip(p["xhat_raw"], corr_vec)]
            e_applied = math.sqrt(sum((o - x) ** 2
                                      for o, x in zip(obs_next,
                                                      xhat_applied)))
            benefit_shadow = e0_raw_abs - e_applied
            self.shadow_usefulness.update(p["qfeat"], benefit_shadow)
            self._shadow_hist.append((p["uhat_gate"], benefit_shadow))

        self._tick += 1
        self._pending = None

    # -- direct transition interface (for shuffle/replay probes) ------------
    def learn_transition(self, obs_vec: List[float], action: int,
                         obs_next: List[float], reward: float) -> float:
        """Bare single-transition training: model.observe + memory store.

        No retrieval correction, no affect, no prediction logging — the
        minimal learning path, so K5 (shuffle) isolates temporal-structure
        learning in the weights/precision machinery. Returns |e0|.
        """
        ctx = self.context_key(obs_vec)
        xhat = self.model.predict_next(obs_vec, action, ctx)
        diag = self.model.observe(obs_vec, action, ctx, obs_next, reward,
                                  lesion_l1=self.lesion_l1)
        e0 = [o - x for o, x in zip(obs_next, xhat)]
        e0_abs = math.sqrt(sum(e * e for e in e0))
        attention = max(0.0, min(1.0, e0_abs))
        self.memory.store(obs_vec, action, reward, e0, diag["e1"],
                          diag["pi0"], attention, self._tick, self._episode,
                          self.env_name)
        self._tick += 1
        return e0_abs

    def eval_transition(self, obs_vec: List[float], action: int,
                        obs_next: List[float]) -> float:
        """Prediction error |e0| for one transition, no learning."""
        ctx = self.context_key(obs_vec)
        xhat = self.model.predict_next(obs_vec, action, ctx)
        return math.sqrt(sum((o - x) ** 2
                             for o, x in zip(obs_next, xhat)))

    # -- persistence ---------------------------------------------------------
    def snapshot(self) -> Dict:
        return {
            "config": {
                "observation_space": self.observation_space,
                "n_actions": self.n_actions, "env_name": self.env_name,
                "action_mode": self.action_mode,
                "affect": self.affect_kind,
                "lesion_l1": self.lesion_l1,
                "uniform_precision": self.uniform_precision,
                "memory_enabled": self.memory_enabled,
                "precision_kind": self.precision_kind,
                "surprise_k": self.surprise_k,
                "frozen": self.frozen, "seed": self.seed,
            },
            "model": self.model.snapshot(),
            "memory": self.memory.snapshot(),
            "selector": self.selector.snapshot(),
            "err_affect": self.err_affect.snapshot(),
            "pad": self.pad.snapshot(),
            "usefulness": self.usefulness.snapshot(),
            "shadow_usefulness": self.shadow_usefulness.snapshot(),
            "gate_uhat_source": self.gate_uhat_source,
            "gate": self.gate.snapshot(),
            "episode": self._episode, "tick": self._tick,
            "last_action": self._last_action, "e1_ema": self._e1_ema,
            "knobs": self.knobs,
            "rng_state": self._rng.getstate(),
        }

    def restore(self, state: Dict) -> None:
        cfg = state["config"]
        if (cfg["n_actions"] != self.n_actions
                or cfg["observation_space"] != self.observation_space):
            raise ValueError("agent config mismatch on restore")
        self.model.restore(state["model"])
        self.memory.restore(state["memory"])
        self.selector.restore(state["selector"])
        self.err_affect.restore(state["err_affect"])
        self.pad.restore(state["pad"])
        self.usefulness.restore(state["usefulness"])
        # Tolerant of pre-shadow snapshots (gate_uhat_source defaults live).
        self.gate_uhat_source = state.get("gate_uhat_source", "live")
        su = state.get("shadow_usefulness")
        if su is not None:
            self.shadow_usefulness.restore(su)
        self.gate.restore(state["gate"])
        self._episode = state["episode"]
        self._tick = state["tick"]
        self._last_action = state["last_action"]
        self._e1_ema = state["e1_ema"]
        self.knobs = state["knobs"]
        self._rng.setstate(state["rng_state"])
        self._pending = None
