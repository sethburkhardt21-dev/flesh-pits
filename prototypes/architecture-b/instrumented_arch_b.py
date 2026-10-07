"""Additive prediction-time instrumentation for Architecture B — EXP-FP-0071.

InstrumentedArchB subclasses ArchB WITHOUT modifying agent.py or any core
file. Behavior is byte-identical to ArchB: _close_tick() captures the
prediction-time context from self._pending BEFORE delegating to super(),
then appends it to a sidecar JSONL (one line per tick, keyed by
tick/episode).

Captured context (all known to the agent BEFORE the tick's outcome):
  mean_similarity, n_nbrs  -- episodic-memory retrieval match quality
  retrieval_used            -- whether the memory correction was applied
  ounc, runc, uunc, uhat   -- stated uncertainties / usefulness estimate
  e1_ema                  -- L1 error-scale EMA at prediction time

These are the per-tick novelty/state signals the variant-A estimator
lacked: EXP-FP-0070's negative showed B's changing_rule error magnitude is
nearly temporally i.i.d. (lag-1 autocorr 0.13), so error HISTORY carries
almost no signal; the predictable component must come from the current
tick's context (novelty -> error).

Deterministic, stdlib only. The sidecar write is observational only.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ArchB  # noqa: E402


class InstrumentedArchB(ArchB):
    NAME = "arch_b_instrumented"

    def __init__(self, *args, sidecar_path: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sidecar_path = sidecar_path
        self._sidecar_fh = None
        if sidecar_path:
            self._sidecar_fh = open(sidecar_path, "w")

    def _close_tick(self, obs_next) -> None:
        p = self._pending
        assert p is not None
        ctx = {"tick": p["tick"], "episode": p["episode"],
               "mean_similarity": float(p["mean_similarity"]),
               "n_nbrs": int(p["n_nbrs"]),
               "retrieval_used": bool(p["retrieval_used"]),
               "ounc": float(p["ounc"]), "runc": float(p["runc"]),
               "uunc": float(p["uunc"]), "uhat": float(p["uhat"]),
               "e1_ema": float(self._e1_ema)}
        super()._close_tick(obs_next)
        if self._sidecar_fh is not None:
            self._sidecar_fh.write(json.dumps(ctx, sort_keys=True) + "\n")

    def close_sidecar(self) -> None:
        if self._sidecar_fh is not None:
            self._sidecar_fh.close()
            self._sidecar_fh = None
