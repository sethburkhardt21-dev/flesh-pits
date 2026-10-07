"""pomaze_dense — additive dense-reward probe variant of pomaze.

NAME: pomaze_dense | VERSION: 1.0.0 | PARENT: pomaze v1.0.0

Additive ONLY: subclasses POMaze and overrides step() to add a shaping
term +0.02 * beacon per step (beacon is the documented coarse
goal-proximity gradient in {0, 1/3, 2/3, 1}). Everything else — maze
generation, start/goal, step/bump penalties (-0.01/-0.02), +1.0 goal
reward, horizon (200), partial observation channels — is inherited
unchanged. Canonical pomaze.py is NOT modified.

Purpose (EXP-FP-0010-POMAZE-DIAG): discriminate whether the pomaze
failure is reward-findability (exploration) vs inability to use
available signal (representation). If an architecture solves the dense
variant but not the canonical env, the failure is sparsity, not
navigation machinery. If it still fails with a dense gradient, the
failure is representational.

The shaping is deliberately near-linear in the observation (beacon is an
obs channel): a linear reward head CAN learn it. That is the point — the
probe separates "cannot find reward" from "cannot learn this mapping".

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pomaze import POMaze  # noqa: E402


class POMazeDense(POMaze):
    NAME = "pomaze_dense"
    VERSION = "1.0.0"
    SHAPING_ALPHA = 0.02  # per-step shaping weight on the beacon channel

    def step(self, action: int):
        obs, reward, done, info = super().step(action)
        # Shaping added AFTER the goal check: reaching the goal still
        # yields +1.0 (plus the final step's shaping).
        reward = reward + self.SHAPING_ALPHA * obs["beacon"]
        info = dict(info)
        info["shaping"] = self.SHAPING_ALPHA * obs["beacon"]
        return obs, reward, done, info
