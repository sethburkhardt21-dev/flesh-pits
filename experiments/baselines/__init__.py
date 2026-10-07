"""The §39 baseline agents. One import per baseline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stateless_agent import StatelessAgent  # noqa: E402
from persistent_no_learning import PersistentNoLearningAgent  # noqa: E402
from fixed_attention import FixedAttentionAgent  # noqa: E402
from no_episodic_memory import NoEpisodicMemoryAgent  # noqa: E402
from fixed_predictor import FixedPredictorAgent  # noqa: E402
from no_metacognition import NoMetacognitionAgent  # noqa: E402
from symbolic_baseline import SymbolicBaselineAgent  # noqa: E402

ALL_BASELINES = [StatelessAgent, PersistentNoLearningAgent, FixedAttentionAgent,
                 NoEpisodicMemoryAgent, FixedPredictorAgent,
                 NoMetacognitionAgent, SymbolicBaselineAgent]

__all__ = ["StatelessAgent", "PersistentNoLearningAgent", "FixedAttentionAgent",
           "NoEpisodicMemoryAgent", "FixedPredictorAgent",
           "NoMetacognitionAgent", "SymbolicBaselineAgent", "ALL_BASELINES"]
