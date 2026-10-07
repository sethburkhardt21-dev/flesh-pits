"""Persistent-state-but-no-learning agent (§39).

Keeps persistent state across steps AND episodes (transition log, visitation
counts) — but the policy is FIXED: a random linear map over the flattened
observation, weights drawn once from a constant seed, never updated.
The log exists for audit; act() never reads it.

This isolates "having state" from "learning from experience": if a learning
agent beats this one, the gain comes from adaptation, not mere persistence.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, flatten_obs, argmax_det, new_rng  # noqa: E402


class PersistentNoLearningAgent(BaselineAgent):
    NAME = "persistent_no_learning"

    POLICY_SEED = 0xC0FFEE  # fixed: identical policy every episode and run

    def _on_reset(self):
        self._log = []        # persistent transition record (audit only)
        self._visits = {}     # obs_key -> count (audit only)
        self._W = None        # fixed random linear policy, lazily sized

    def _ensure_policy(self, dim):
        if self._W is None:
            rng = new_rng(self.POLICY_SEED)
            self._W = [[rng.uniform(-1.0, 1.0) for _ in range(self._n_actions)]
                       for _ in range(dim)]

    def _key(self, obs):
        return tuple(round(x, 3) for x in flatten_obs(obs))

    def act(self, obs: dict) -> int:
        v = flatten_obs(obs)
        self._ensure_policy(len(v))
        key = self._key(obs)
        self._visits[key] = self._visits.get(key, 0) + 1
        scores = [sum(v[i] * self._W[i][a] for i in range(len(v)))
                  for a in range(self._n_actions)]
        return argmax_det(scores)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        # Recorded, never used for decisions: persistence without learning.
        self._log.append((self._key(obs), action, reward))

    def _extra_state(self):
        return {"visits": {str(k): c for k, c in self._visits.items()},
                "log_len": len(self._log)}

    def _restore_extra(self, state):
        self._visits = {eval(k): c for k, c in state["visits"].items()}
        self._W = None  # re-derived deterministically on next act()
