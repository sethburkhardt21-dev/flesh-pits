"""Retrieval-correction gating — Architecture B (§9 item 6, B gap #2).

The UsefulnessPredictor (predictions.py) computes uhat = predicted
error-reduction from applying the memory retrieval correction, but
ArchB._open_tick has always applied the correction unconditionally.
This module decides per tick whether to APPLY an available correction:

  policy "ungated": always apply. This is the CURRENT ArchB behavior;
                    the default, so existing code paths are bit-identical
                    when the gate is not engaged.
  policy "uhat":    apply iff uhat > threshold. The decision boundary is
                    the sign of the predicted benefit (preregistered
                    threshold=0.0: apply only when the predicted benefit
                    is positive).
  policy "random":  apply with Bernoulli(rate) per tick via a seeded RNG.
                    The chance control: random gating at the same
                    application rate as the uhat arm (does uhat beat
                    chance?).

The gate NEVER touches the predictor: it only READS uhat, exactly as
predicted by predictions.UsefulnessPredictor. No learning-rate changes,
no predictor surgery, no re-tuning.

Diagnostics count only ticks where a correction was actually available
(decisions are moot when correction is None). `stats()` reports the
application rate, the uhat mean for applied vs blocked decisions, and
counters. Snapshot/restore cover the RNG state and counters.

Deterministic. Stdlib only.
"""

from __future__ import annotations

import random
from typing import Dict, List


class RetrievalGate:
    """Per-tick apply/skip decision for an available retrieval correction."""

    POLICY_UNGATED = "ungated"
    POLICY_UHAT = "uhat"
    POLICY_RANDOM = "random"
    POLICY_P_EXT = "p_ext"

    def __init__(self, policy: str = "ungated", threshold: float = 0.0,
                 rate: float = 1.0, seed: int = 0) -> None:
        if policy not in (self.POLICY_UNGATED, self.POLICY_UHAT,
                          self.POLICY_RANDOM, self.POLICY_P_EXT):
            raise ValueError(f"unknown retrieval-gate policy: {policy!r}")
        if policy == self.POLICY_RANDOM and not 0.0 <= rate <= 1.0:
            raise ValueError("random-gate rate must be in [0, 1]")
        self.policy = policy
        self.threshold = float(threshold)
        self.rate = float(rate)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        # External probability for POLICY_P_EXT (Track D decision-use tests):
        # the driver sets gate._p_ext before each act(); decide() applies iff
        # _p_ext > threshold. Default 0.5 = abstain-neutral.
        self._p_ext = 0.5
        # Diagnostics (ticks where a correction was available only).
        self.n_available = 0
        self.n_applied = 0
        self._uhat_sum = 0.0
        self._uhat_applied_sum = 0.0
        self._uhat_blocked_sum = 0.0

    def decide(self, uhat: float) -> bool:
        """Return True iff the available correction should be applied.

        MUST be called exactly once per tick when (and only when) a
        correction is available, so the counters stay honest.
        """
        uhat = float(uhat)
        if self.policy == self.POLICY_UHAT:
            apply = uhat > self.threshold
        elif self.policy == self.POLICY_RANDOM:
            apply = self._rng.random() < self.rate
        else:  # "ungated"
            apply = True
        self.n_available += 1
        self._uhat_sum += uhat
        if apply:
            self.n_applied += 1
            self._uhat_applied_sum += uhat
        else:
            self._uhat_blocked_sum += uhat
        return apply

    def application_rate(self) -> float:
        if self.n_available == 0:
            return 0.0
        return self.n_applied / self.n_available

    def stats(self) -> Dict:
        n_app = self.n_applied
        n_blk = self.n_available - self.n_applied
        return {
            "policy": self.policy,
            "threshold": self.threshold,
            "rate_param": self.rate,
            "n_available": self.n_available,
            "n_applied": n_app,
            "n_blocked": n_blk,
            "application_rate": self.application_rate(),
            "mean_uhat": (self._uhat_sum / self.n_available
                          if self.n_available else None),
            "mean_uhat_applied": (self._uhat_applied_sum / n_app
                                   if n_app else None),
            "mean_uhat_blocked": (self._uhat_blocked_sum / n_blk
                                   if n_blk else None),
        }

    def snapshot(self) -> Dict:
        return {
            "policy": self.policy,
            "threshold": self.threshold,
            "rate": self.rate,
            "seed": self.seed,
            "rng_state": list(self._rng.getstate()),
            "n_available": self.n_available,
            "n_applied": self.n_applied,
            "uhat_sums": [self._uhat_sum, self._uhat_applied_sum,
                          self._uhat_blocked_sum],
        }

    def restore(self, state: Dict) -> None:
        if state["policy"] != self.policy:
            raise ValueError("gate policy mismatch on restore")
        self.threshold = state["threshold"]
        self.rate = state["rate"]
        self.seed = state["seed"]
        rng_state = state["rng_state"]
        if isinstance(rng_state, list):
            ver, internal, gauss_next = rng_state
            self._rng.setstate((ver, tuple(internal), gauss_next))
        else:
            self._rng.setstate(rng_state)
        self.n_available = state["n_available"]
        self.n_applied = state["n_applied"]
        (self._uhat_sum, self._uhat_applied_sum,
         self._uhat_blocked_sum) = state["uhat_sums"]
