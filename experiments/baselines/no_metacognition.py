"""No-metacognition agent (§39).

Tabular Q-learning (TD(0)) on discretized observations. Learns action
values from experience — but has NO uncertainty estimation, NO confidence
scores, NO exploration bonus: exploration is a FIXED epsilon schedule
(epsilon=0.10, seeded), never driven by uncertainty.

This is the baseline metacognitive mechanisms must beat: if adding
calibrated confidence / uncertainty-driven exploration cannot outperform
fixed-epsilon Q-learning, the metacognition is decorative.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, TransitionLearner, flatten_obs, argmax_det  # noqa: E402


class NoMetacognitionAgent(BaselineAgent, TransitionLearner):
    NAME = "no_metacognition"

    ALPHA = 0.2
    GAMMA = 0.95
    EPSILON = 0.10  # fixed schedule; NOT uncertainty-driven

    def _on_reset(self):
        self._tl_reset()
        if not hasattr(self, "_Q"):
            self._Q = {}  # (obs_key, action) -> value; persists across episodes

    def _key(self, obs_vec):
        return tuple(round(x, 1) for x in obs_vec)

    def _q(self, key, a):
        return self._Q.get((key, a), 0.0)

    def act(self, obs: dict) -> int:
        v = flatten_obs(obs)
        key = self._key(v)
        if self._rng.random() < self.EPSILON:
            return self._rng.randrange(self._n_actions)
        return argmax_det([self._q(key, a) for a in range(self._n_actions)])

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        v = flatten_obs(obs)
        self._tl_update(v, action, reward, done, len(v))

    def learn_transition(self, obs_vec, action, next_obs_vec, reward, done):
        key = self._key(obs_vec)
        nkey = self._key(next_obs_vec)
        target = reward
        if not done:
            target += self.GAMMA * max(self._q(nkey, a)
                                       for a in range(self._n_actions))
        old = self._q(key, action)
        self._Q[(key, action)] = old + self.ALPHA * (target - old)

    def _extra_state(self):
        s = self._tl_state()
        s["Q"] = [[list(k[0]), k[1], v] for k, v in self._Q.items()]
        return s

    def _restore_extra(self, state):
        self._tl_restore(state)
        self._Q = {(tuple(e[0]), e[1]): e[2] for e in state["Q"]}
