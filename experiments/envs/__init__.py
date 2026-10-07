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
from delayed_multistep import DelayedMultistep  # noqa: E402
from compositional_rule import CompositionalRule  # noqa: E402
from self_world import SelfWorld  # noqa: E402
from self_world_cl import SelfWorldCL  # noqa: E402

ALL_ENVS = [GridWorld, POMaze, ChangingRule, DelayedReward, ResourceWorld,
            CueDelayedReward, DelayedMultistep, CompositionalRule, SelfWorld,
            SelfWorldCL]

__all__ = ["GridWorld", "POMaze", "ChangingRule", "DelayedReward",
           "ResourceWorld", "CueDelayedReward", "DelayedMultistep",
           "CompositionalRule", "SelfWorld", "SelfWorldCL", "ALL_ENVS"]
