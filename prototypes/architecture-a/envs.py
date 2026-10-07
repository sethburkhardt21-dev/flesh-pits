"""Architecture A — minimal closed-loop environments (PROVISIONAL).

The parallel environments worker is building the canonical ENV_INTERFACE
(flesh-pits/experiments/ENV_INTERFACE.md); it is NOT on disk yet as of
2026-10-07. These local environments exist only so Architecture A's
learning claims (K4) are testable in closed loop NOW, per the directive's
"kill continuously ... from the first build" rule. When the canonical
spec lands, K4 will be re-run against it and this file marked
SUPERSEDED or conformed.

All environments: deterministic under an explicit seed, stdlib only,
no identity machinery.

ChangingRelevanceEnv (K4 task):
  n channels; latent relevance weights w sum to 1. At tick T/2 the
  weights REVERSE (relevance flips: the attended channel becomes
  worthless). Observation per channel: noisy signal of its relevance.
  Reward for action a: w_a + noise. A fixed-attention agent keeps
  attending by raw salience; a learned-attention agent must re-weight
  its gains to track the reversal.
"""
from __future__ import annotations

import math
import random


class ChangingRelevanceEnv:
    def __init__(self, channels, total_ticks, *, seed=0, noise=0.05,
                 stationary_signals=False):
        """stationary_signals=True: observations are identical across
        channels and constant over time (0.5 + noise) while reward
        weights flip at T/2. This is the K4 discrimination condition:
        salience carries NO information about relevance, so only
        reward-driven learned attention can track the reversal."""
        self.channels = list(channels)
        n = len(self.channels)
        self.total_ticks = int(total_ticks)
        self.rng = random.Random(seed)
        self.noise = float(noise)
        self.stationary_signals = bool(stationary_signals)
        self.tick = 0
        # phase 1: relevance concentrated on channel 0; phase 2: reversed
        w1 = [0.7] + [0.3 / (n - 1)] * (n - 1)
        w2 = list(reversed(w1))
        self.weights_phase = (w1, w2)

    def weights(self):
        phase = 0 if self.tick < self.total_ticks // 2 else 1
        return dict(zip(self.channels, self.weights_phase[phase]))

    def observe(self):
        if self.stationary_signals:
            return {c: max(0.0, min(1.0, 0.5 + self.rng.gauss(0, self.noise)))
                    for c in self.channels}
        w = self.weights()
        return {c: max(0.0, min(1.0, w[c] + self.rng.gauss(0, self.noise)))
                for c in self.channels}

    def step(self, action):
        w = self.weights()
        r = w.get(action, 0.0) + self.rng.gauss(0, self.noise)
        self.tick += 1
        return max(0.0, min(1.0, r))


def make_specialists(channels):
    """Specialists: stimulus = observed channel signal; payload is neutral."""
    specs = []
    for c in channels:
        def fn(obs, tick, _c=c):
            return float(obs.get(_c, 0.0)), {"summary": f"signal:{_c}",
                                             "signal": float(obs.get(_c, 0.0))}
        specs.append((c, fn))
    return specs
