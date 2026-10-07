"""The §35 closed-loop environments. One import per environment."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid_world import GridWorld  # noqa: E402
from pomaze import POMaze  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402
from delayed_reward import DelayedReward  # noqa: E402
from resource_world import ResourceWorld  # noqa: E402
from cue_delayed_reward import CueDelayedReward  # noqa: E402

ALL_ENVS = [GridWorld, POMaze, ChangingRule, DelayedReward, ResourceWorld,
            CueDelayedReward]

__all__ = ["GridWorld", "POMaze", "ChangingRule", "DelayedReward",
           "ResourceWorld", "CueDelayedReward", "ALL_ENVS"]
