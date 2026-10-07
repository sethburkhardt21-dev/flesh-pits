"""Fixed-predictor agent (§39).

Same predictor SHAPE as NoEpisodicMemoryAgent, but weights are RANDOM and
FROZEN (drawn once from a constant seed, never updated). Acts greedily on
its (uninformative) predicted reward.

Isolates the value of predictor LEARNING: if the learning version beats
this one, the predictor's adaptation — not its architecture — earned it.
(Architecture B kill experiment K2: learned vs. fixed predictor.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, flatten_obs, argmax_det, new_rng  # noqa: E402
from predictor import LinearPredictor  # noqa: E402


class FixedPredictorAgent(BaselineAgent):
    NAME = "fixed_predictor"

    FROZEN_SEED = 0xF1E07  # weights frozen at these random values forever

    def _on_reset(self):
        self._pred = None
        self._frng = new_rng(self.FROZEN_SEED)

    def _ensure_pred(self, obs):
        v = flatten_obs(obs)
        if self._pred is None:
            # lr=0.0 AND update() never called: doubly frozen, honestly static.
            self._pred = LinearPredictor(len(v), self._n_actions, 0.0, self._frng)
        return v

    def act(self, obs: dict) -> int:
        v = self._ensure_pred(obs)
        scores = [self._pred.predict(v, a)[1] for a in range(self._n_actions)]
        return argmax_det(scores)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        pass  # frozen: no learning, by design
