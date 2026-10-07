"""Comparator-based source attribution instrument (additive).

ComparatorAgent: the Row-10 computational analogue — efference-copy tagging
of every action; forward-model prediction of sensory consequences;
comparator residual -> externally-caused attribution. The forward model is
LEARNED online per channel (action->delta coupling, EMA), not hand-wired to
channel names: the instrument must discover which channel its actions drive.

Architecture note (2026-10-07): the driver contract calls
act(pre_obs) -> step -> update(pre_obs, ...). The POST-step observation is
only visible at the NEXT act() call, so each completed transition is
processed inside act() when the next observation arrives. update() is a
no-op hook (kept for the Agent interface). Each processed transition covers
action tick k (action a_k, pre obs_k, post obs_{k+1}).

On each moved channel (|delta| > STEP/2):
    residual = |actual_channel - predicted_channel|
    attribution = "world" if residual > STEP/4 else "self"

predicted_channel = reflect(channel_t + pred_delta), pred_delta =
action_sign*STEP when |coupling| > 0.5 else 0.0.

ChanceAgent: the preregistered null — random 50/50 attribution per moved
channel from a derived RNG stream.

Both agents emit per-tick forward predictions tagged "predicted" for the
quarantine test (the prediction buffer lives outside the store; admission
of predicted records is tested separately in episodic_store).

Stdlib only. No random.* module calls (env_interface RNG only).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "envs"))

from env_interface import Agent, new_rng  # noqa: E402

STEP = 0.2
TOL = STEP / 4.0
MOVE_THRESH = STEP / 2.0
COUPLING_ETA = 0.1
COUPLING_GATE = 0.5
BURN_IN = 8


def reflect(v: float) -> float:
    if v < 0.0:
        return -v
    if v > 1.0:
        return 2.0 - v
    return v


class ComparatorAgent(Agent):
    """Efference-copy + learned forward-model comparator."""

    NAME = "comparator_attributor"

    def __init__(self):
        self._channels = None
        self._coupling = {}
        self._pending = None       # (pre_obs, action) awaiting post obs
        self._step_count = 0       # ticks so far (action ticks are 1-indexed)
        self.attributions = []     # list of {"tick": k, "attribution": {...}}
        self.predictions = []      # forward predictions tagged "predicted"

    # -- Agent API ------------------------------------------------------
    def reset(self, seed: int, action_space: dict) -> None:
        self._seed = int(seed)
        self._channels = None
        self._coupling = {}
        self._pending = None
        self._step_count = 0
        self.attributions = []
        self.predictions = []

    def act(self, obs: dict) -> int:
        if self._channels is None:
            self._channels = sorted(obs.keys())
            self._coupling = {c: 0.0 for c in self._channels}
        if self._pending is not None:
            pre, action = self._pending
            self._process_transition(pre, obs, action, self._step_count)
            self._pending = None
        # Preregistered policy: deterministic alternation 0,1,0,1,...
        action = self._step_count % 2
        self._pending = (dict(obs), action)
        self._step_count += 1
        # Emit forward prediction for the coming tick (world-model head).
        next_sign = 1.0 if action == 1 else -1.0
        self.predictions.append({
            "tick": self._step_count + 1,
            "source": "predicted",
            "observation": {ch: self._predict(ch, obs[ch], next_sign)
                            for ch in self._channels},
        })
        return action

    def _process_transition(self, pre, post, action, tick):
        action_sign = 1.0 if action == 1 else -1.0
        for ch in self._channels:
            d = post[ch] - pre[ch]
            if abs(d) > 1e-12:
                s = 1.0 if (d > 0.0) == (action_sign > 0.0) else -1.0
                self._coupling[ch] += COUPLING_ETA * (s - self._coupling[ch])
        if tick > BURN_IN:
            rec = {"tick": tick, "attribution": {}}
            for ch in self._channels:
                d = post[ch] - pre[ch]
                if abs(d) <= MOVE_THRESH:
                    continue
                pred = self._predict(ch, pre[ch], action_sign)
                residual = abs(post[ch] - pred)
                rec["attribution"][ch] = ("world" if residual > TOL
                                          else "self")
            self.attributions.append(rec)

    def _predict(self, ch: str, ch_t: float, action_sign: float) -> float:
        if abs(self._coupling.get(ch, 0.0)) > COUPLING_GATE:
            sgn = 1.0 if self._coupling[ch] > 0.0 else -1.0
            return reflect(ch_t + sgn * action_sign * STEP)
        return ch_t  # no learned coupling: predict stasis

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        # Transition learning happens in act() when the post-step observation
        # arrives (see module docstring). info is never read (contract §5).
        return None

    def snapshot(self) -> dict:
        return {"coupling": dict(self._coupling),
                "step_count": self._step_count, "seed": self._seed}

    def restore(self, state: dict) -> None:
        self._coupling = dict(state["coupling"])
        self._step_count = state["step_count"]
        self._seed = state["seed"]


class ChanceAgent(Agent):
    """Preregistered null: random 50/50 attribution per moved channel."""

    NAME = "chance_attributor"

    def __init__(self):
        self._channels = None
        self._pending = None
        self._step_count = 0
        self.attributions = []

    def reset(self, seed: int, action_space: dict) -> None:
        self._rng = new_rng(int(seed) ^ 0x5EED)
        self._channels = None
        self._pending = None
        self._step_count = 0
        self.attributions = []

    def act(self, obs: dict) -> int:
        if self._channels is None:
            self._channels = sorted(obs.keys())
        if self._pending is not None:
            pre, action = self._pending
            self._process_transition(pre, obs, action, self._step_count)
            self._pending = None
        action = self._step_count % 2  # same action stream as comparator
        self._pending = (dict(obs), action)
        self._step_count += 1
        return action

    def _process_transition(self, pre, post, action, tick):
        if tick > BURN_IN:
            rec = {"tick": tick, "attribution": {}}
            for ch in self._channels:
                d = post[ch] - pre[ch]
                if abs(d) <= MOVE_THRESH:
                    continue
                rec["attribution"][ch] = ("world" if self._rng.random() < 0.5
                                          else "self")
            self.attributions.append(rec)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        return None

    def snapshot(self) -> dict:
        return {"step_count": self._step_count}

    def restore(self, state: dict) -> None:
        self._step_count = state["step_count"]
