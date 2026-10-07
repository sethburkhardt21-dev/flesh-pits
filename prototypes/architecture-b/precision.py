"""Precision from error statistics — Architecture B.

Precision pi_l is ESTIMATED from recent prediction-error statistics, not
hand-set. For each level l and channel c we keep a running window of recent
errors e_l and set::

    pi_lc = 1 / (var_lc + eps)

(clipped to [pi_min, pi_max]). Noisy channels are down-weighted; reliable
channels drive larger belief updates. The belief update then follows::

    dmu_l propto pi_l * e_l

Ablation hook: ``PrecisionEstimator(uniform=True)`` (or ``set_uniform(True)``)
forces every pi to exactly 1.0, so the precision machinery can be removed
from the causal path and its contribution measured (load-bearing claim 2).

Deterministic: no RNG anywhere. Stdlib only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List


class PrecisionEstimator:
    """Per-channel precision from a sliding window of recent errors."""

    def __init__(self, n_channels: int, window: int = 50,
                 eps: float = 1e-6, pi_min: float = 1e-2,
                 pi_max: float = 20.0, uniform: bool = False) -> None:
        # pi_max is a stability safety clip: with the update dW = eta*pi*e,
        # unbounded pi lets a single surprising tick detonate the weights.
        # pi is still ESTIMATED from error statistics below the clip.
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if window < 2:
            raise ValueError("window must be >= 2")
        self.n_channels = n_channels
        self.window = window
        self.eps = eps
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.uniform = uniform
        self._hist: List[deque] = [deque(maxlen=window)
                                   for _ in range(n_channels)]
        self._last_pi: List[float] = [1.0] * n_channels

    # -- core -----------------------------------------------------------
    def observe(self, errors: List[float]) -> List[float]:
        """Record one error vector; return the current precision vector.

        With uniform=True every entry is exactly 1.0 (ablation).
        """
        if len(errors) != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} errors, got {len(errors)}")
        if self.uniform:
            self._last_pi = [1.0] * self.n_channels
            return list(self._last_pi)
        for c, e in enumerate(errors):
            self._hist[c].append(float(e))
        pi = []
        for c in range(self.n_channels):
            h = self._hist[c]
            if len(h) < 2:
                pi.append(1.0)
                continue
            mean = sum(h) / len(h)
            var = sum((x - mean) ** 2 for x in h) / (len(h) - 1)
            p = 1.0 / (var + self.eps)
            pi.append(max(self.pi_min, min(self.pi_max, p)))
        self._last_pi = pi
        return list(pi)

    def current(self) -> List[float]:
        """Most recently computed precision vector (1.0s if uniform)."""
        if self.uniform:
            return [1.0] * self.n_channels
        return list(self._last_pi)

    def set_uniform(self, flag: bool) -> None:
        """Ablation switch: True forces all pi = 1."""
        self.uniform = bool(flag)

    def channel_variance(self) -> List[float]:
        """Running per-channel error variance (diagnostic)."""
        out = []
        for h in self._hist:
            if len(h) < 2:
                out.append(0.0)
                continue
            mean = sum(h) / len(h)
            out.append(sum((x - mean) ** 2 for x in h) / (len(h) - 1))
        return out

    # -- persistence ----------------------------------------------------
    def snapshot(self) -> Dict:
        return {
            "n_channels": self.n_channels,
            "window": self.window,
            "eps": self.eps,
            "pi_min": self.pi_min,
            "pi_max": self.pi_max,
            "uniform": self.uniform,
            "hist": [list(h) for h in self._hist],
            "last_pi": list(self._last_pi),
        }

    def restore(self, state: Dict) -> None:
        if state["n_channels"] != self.n_channels:
            raise ValueError("precision channel mismatch on restore")
        self.window = state["window"]
        self.eps = state["eps"]
        self.pi_min = state["pi_min"]
        self.pi_max = state["pi_max"]
        self.uniform = state["uniform"]
        self._hist = [deque(h, maxlen=self.window)
                      for h in state["hist"]]
        self._last_pi = list(state["last_pi"])
