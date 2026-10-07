"""Architecture A — salience bidding (change + habituation), z-score based.

Algorithmic port of the donor mechanism (change_bid.py /
habituation_bid.py in the sandbox): sigmoid of a causal running z-score
against the module's own EMA baseline. Change-detection and
habituation-desaturation are the SAME mechanism here (a constant input
bids at the midpoint 0.5 instead of saturating) — one implementation,
two roles, stated honestly.

Stdlib only. Deterministic. No identity machinery anywhere.
"""
from __future__ import annotations

import math

EMA_ALPHA = 0.01
SD_FLOOR = 1e-9


def stable_sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    if z < -700.0:
        return 0.0
    e = math.exp(z)
    return e / (1.0 + e)


class NonFiniteBidRefused(ValueError):
    """Fail-closed intake: non-finite bids refused before any state mutates."""


class RunningZScoreBid:
    """sigmoid((x - running mean) / running sd); statistics read BEFORE update.

    - First reading bids 0.5 (no baseline yet).
    - sd at/below SD_FLOOR -> z = 0 (constant input never amplified).
    - Sustained level shift: baseline follows at rate alpha, bids relax to 0.5.
    """

    def __init__(self, alpha: float = EMA_ALPHA, initial_var: float = 1.0):
        alpha = float(alpha)
        initial_var = float(initial_var)
        if not math.isfinite(alpha) or not math.isfinite(initial_var):
            raise NonFiniteBidRefused("non-finite bid parameters refused")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if initial_var < 0.0:
            raise ValueError("initial_var must be >= 0")
        self.alpha = alpha
        self.initial_var = initial_var
        self.mean = 0.0
        self.var = initial_var
        self.seen = 0

    def bid(self, value: float) -> float:
        v = float(value)
        if not math.isfinite(v):
            raise NonFiniteBidRefused(
                f"refusing non-finite bid value {v!r}; state unchanged")
        z = self._z(v)
        self._update(v)
        return stable_sigmoid(z)

    def _z(self, value: float) -> float:
        if self.seen == 0:
            return 0.0
        sd = math.sqrt(self.var)
        if sd <= SD_FLOOR:
            return 0.0
        return (value - self.mean) / (sd + 1e-12)

    def _update(self, value: float) -> None:
        if self.seen == 0:
            new_mean, new_var = value, self.initial_var
        else:
            delta = value - self.mean
            new_mean = self.mean + self.alpha * delta
            new_var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)
            if not (math.isfinite(delta) and math.isfinite(new_mean)
                    and math.isfinite(new_var)):
                raise NonFiniteBidRefused(
                    f"refusing bid {value!r}: would overflow baseline")
        self.mean, self.var = new_mean, new_var
        self.seen += 1

    @property
    def baseline(self):
        return (self.mean, self.var, self.seen)
