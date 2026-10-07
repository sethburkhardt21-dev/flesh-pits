"""pomaze_near — additive near-goal-start probe variant of pomaze.

NAME: pomaze_near | VERSION: 1.0.0 | PARENT: pomaze v1.0.0

Additive ONLY: subclasses POMaze and overrides reset() to resample the
start position from walkable cells with Manhattan distance <= 4 from the
goal (SIZE-2, SIZE-2) after canonical maze generation. Everything else —
maze generation, rewards, horizon, observations — is inherited unchanged.
Canonical pomaze.py is NOT modified.

Purpose (EXP-FP-0010-POMAZE-DIAG): make the +1.0 reward FREQUENT so the
diagnostic can ask whether each architecture's machinery REGISTERS found
rewards (credit assignment) and whether post-hit episodes improve
(exploitation). If an architecture cannot exploit frequent rewards, its
failure is not merely exploration.

The resample uses the env's own seeded RNG (self._rng), so the variant is
deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pomaze import POMaze  # noqa: E402


class POMazeNear(POMaze):
    NAME = "pomaze_near"
    VERSION = "1.0.0"
    MAX_START_DIST = 4  # Manhattan distance from goal, inclusive

    def reset(self, seed: int) -> dict:
        obs = super().reset(seed)
        gx, gy = self._goal
        candidates = [
            (x, y)
            for (x, y) in self._walkable_cells()
            if abs(x - gx) + abs(y - gy) <= self.MAX_START_DIST
        ]
        # The goal cell itself is always a candidate (distance 0); the
        # list is non-empty by construction.
        self._start = self._rng.choice(candidates)
        self._pos = self._start
        return self._obs()

    def _walkable_cells(self):
        n = self.SIZE
        return [(x, y) for y in range(n) for x in range(n)
                if (x, y) not in self._walls]
