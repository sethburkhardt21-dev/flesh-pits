"""No-episodic-memory agent (§39) — ablation counterpart of Architecture D.

Has a LEARNED online linear predictor (same shape as D's predictor) and acts
greedily on predicted immediate reward — but has NO episodic store: no
retrieval of similar past episodes, no surprise-gated admission.

If D beats this agent, the gain is attributable to the episodic memory
system specifically, not to having a learned predictor.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, TransitionLearner, flatten_obs, argmax_det  # noqa: E402
from predictor import LinearPredictor  # noqa: E402


class NoEpisodicMemoryAgent(BaselineAgent, TransitionLearner):
    NAME = "no_episodic_memory"

    LR = 0.05

    def _on_reset(self):
        self._tl_reset()
        # Predictor persists across episodes (this IS the learning);
        # created lazily once obs dim is known.
        if not hasattr(self, "_pred"):
            self._pred = None
        self._dim = None

    def _ensure_pred(self, obs):
        v = flatten_obs(obs)
        if self._pred is None:
            self._dim = len(v)
            self._pred = LinearPredictor(self._dim, self._n_actions,
                                         self.LR, self._rng)
        return v

    def act(self, obs: dict) -> int:
        v = self._ensure_pred(obs)
        scores = [self._pred.predict(v, a)[1] for a in range(self._n_actions)]
        return argmax_det(scores)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        v = self._ensure_pred(obs)
        self._tl_update(v, action, reward, done, self._dim)

    def learn_transition(self, obs_vec, action, next_obs_vec, reward, done):
        self._pred.update(obs_vec, action, next_obs_vec, reward)

    def _extra_state(self):
        s = self._tl_state()
        s["pred"] = None if self._pred is None else self._pred.get_state()
        return s

    def _restore_extra(self, state):
        self._tl_restore(state)
        if state["pred"] is None:
            self._pred = None
        else:
            p = state["pred"]
            self._dim = p["d"]
            self._pred = LinearPredictor(p["d"], p["n"], p["lr"], self._rng)
            self._pred.set_state(p)
