"""Stateless agent — the absolute floor (§39).

No memory between steps. Seeded uniform-random policy. update() is an
explicit no-op. Any architecture that cannot beat this on a task is broken.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent  # noqa: E402


class StatelessAgent(BaselineAgent):
    NAME = "stateless"

    def act(self, obs: dict) -> int:
        return self._rng.randrange(self._n_actions)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        pass  # no learning, by design
