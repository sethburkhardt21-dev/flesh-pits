"""Shift-robust precision estimator — Architecture B (EXP-AB-C2B).

The C2 kill (NR-B-003) found the mechanism of failure: after a rule flip,
the error-statistics window still says "reliable" (high pi) from the
pre-flip regime, so precision-weighted updates OVERSHOOT on the surprising
post-flip errors, while uniform pi=1 takes steady moderate updates and
adapts better.

ShiftRobustPrecisionEstimator is ONE shift-aware variant, preregistered
before testing: a change-point surprise detector gates the window. Each
tick, before updating the window, it computes the worst per-channel
surprise ratio

    s_c = |e_c| / (running_std_c + eps),   s = max_c s_c

against the PRE-update window statistics. If s > surprise_k (default 4.0),
the histories are cleared and this tick returns pi = 1 everywhere — the
post-shift regime is treated as unknown (uniform) instead of "reliable",
so updates cannot overshoot on stale precision. The window then rebuilds
from post-shift data.

This is "precision that discounts post-shift": estimated when stationary,
uniform right after detected change. It must beat uniform on changing_rule
AND not lose to uniform on a stationary noisy task to earn revival;
otherwise it joins the rejected list.

Deterministic: no RNG. Stdlib only.
"""

from __future__ import annotations

import math
from typing import Dict, List

from precision import PrecisionEstimator


class ShiftRobustPrecisionEstimator(PrecisionEstimator):
    """Precision from error statistics with surprise-triggered window reset."""

    def __init__(self, n_channels: int, window: int = 50,
                 eps: float = 1e-6, pi_min: float = 1e-2,
                 pi_max: float = 20.0, uniform: bool = False,
                 surprise_k: float = 4.0) -> None:
        super().__init__(n_channels, window=window, eps=eps,
                         pi_min=pi_min, pi_max=pi_max, uniform=uniform)
        if surprise_k <= 0:
            raise ValueError("surprise_k must be positive")
        self.surprise_k = float(surprise_k)
        self.resets = 0  # diagnostic: how many change-points were detected

    # -- core -----------------------------------------------------------
    def observe(self, errors: List[float]) -> List[float]:
        if len(errors) != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} errors, got {len(errors)}")
        if not self.uniform and self._surprise(errors) > self.surprise_k:
            # Change-point: forget the pre-shift regime. This tick is
            # uniform (unknown precision); the window rebuilds after.
            self.resets += 1
            for h in self._hist:
                h.clear()
            self._last_pi = [1.0] * self.n_channels
            return list(self._last_pi)
        return super().observe(errors)

    def _surprise(self, errors: List[float]) -> float:
        """Worst per-channel |e| / running-std ratio vs the PRE-update
        window. Channels with < 2 samples are skipped (no baseline yet)."""
        worst = 0.0
        for c, e in enumerate(errors):
            if c >= self.n_channels:
                break
            h = self._hist[c]
            if len(h) < 2:
                continue
            mean = sum(h) / len(h)
            var = sum((x - mean) ** 2 for x in h) / (len(h) - 1)
            ratio = abs(float(e)) / (math.sqrt(var) + self.eps)
            if ratio > worst:
                worst = ratio
        return worst

    # -- persistence ----------------------------------------------------
    def snapshot(self) -> Dict:
        snap = super().snapshot()
        snap["surprise_k"] = self.surprise_k
        snap["resets"] = self.resets
        return snap

    def restore(self, state: Dict) -> None:
        super().restore(state)
        self.surprise_k = float(state.get("surprise_k", 4.0))
        self.resets = int(state.get("resets", 0))
