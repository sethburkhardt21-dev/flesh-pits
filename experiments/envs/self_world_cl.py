"""Self/world closed-loop control environment (additive extension).

NAME: self_world_cl | VERSION: 1.0.0

Identical transition dynamics to self_world v1.0.0 — this class SUBCLASSES
SelfWorld and overrides ONLY the reward. The canonical self_world.py is
untouched (same NAME/VERSION, same zero-reward semantics there).

Reward semantics: reward = -(hand - TARGET)^2 with TARGET = 0.5.
Control quality therefore depends on predicting the SELF-caused (hand)
channel: an agent whose model predicts hand transitions well can hold the
hand near target; ball remains WORLD-caused and reward-irrelevant.

info carries the same ground-truth cause labels as self_world, for
SCORING ONLY (contract v1.0.0 §5): {"cause": {"hand": "self",
"ball": "world"}}. Agents are contractually forbidden from using info.

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_world import SelfWorld  # noqa: E402


class SelfWorldCL(SelfWorld):
    NAME = "self_world_cl"
    VERSION = "1.0.0"
    TARGET = 0.5

    def observation_space(self):
        space = super().observation_space()
        space["hand"]["desc"] += (" TASK: reward = -(hand - 0.5)^2 "
                                  "(self_world_cl only).")
        return space

    def step(self, action: int):
        obs, _, done, info = super().step(action)
        reward = -((obs["hand"] - self.TARGET) ** 2)
        return obs, reward, done, info
