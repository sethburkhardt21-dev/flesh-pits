"""Fixed-attention agent (§39).

Maintains a buffer of the last CAP observations. Attention weights are FIXED
and uniform (simple mean over the buffer) — attention does NOT adapt to
novelty, error, or utility. Readout is a fixed random linear map (constant
seed, never updated).

This is the baseline that learned-attention mechanisms (Architecture A K4)
must beat: if adaptive attention cannot outperform uniform averaging here,
the adaptation is decorative.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, flatten_obs, argmax_det, new_rng  # noqa: E402


class FixedAttentionAgent(BaselineAgent):
    NAME = "fixed_attention"

    CAP = 8
    POLICY_SEED = 0xA77E  # fixed readout weights

    def _on_reset(self):
        self._buf = []
        self._W = None

    def _ensure_policy(self, dim):
        if self._W is None:
            rng = new_rng(self.POLICY_SEED)
            self._W = [[rng.uniform(-1.0, 1.0) for _ in range(self._n_actions)]
                       for _ in range(dim)]

    def act(self, obs: dict) -> int:
        v = flatten_obs(obs)
        self._ensure_policy(len(v))
        self._buf.append(v)
        self._buf = self._buf[-self.CAP:]
        # FIXED uniform attention = mean over buffer (no adaptation).
        n, d = len(self._buf), len(v)
        attended = [sum(b[i] for b in self._buf) / n for i in range(d)]
        scores = [sum(attended[i] * self._W[i][a] for i in range(d))
                  for a in range(self._n_actions)]
        return argmax_det(scores)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        pass  # no learning, by design

    def _extra_state(self):
        return {"buf": [list(b) for b in self._buf]}

    def _restore_extra(self, state):
        self._buf = [tuple(b) for b in state["buf"]]
        self._W = None
