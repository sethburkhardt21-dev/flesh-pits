"""Architecture A — attention arbitration with learned-gain hooks.

One arbitration cycle per tick:
  stimulus[channel] -> RunningZScoreBid -> raw bid
  raw bid x learned gain[channel] -> competed bid
  winner = argmax competed bid (canonical channel order tie-break)

LEARNED ATTENTION (K4 hooks, directive §25 real-learning standard):
  - gains start at 1.0 (neutral); `update_gains()` adapts them from
    per-channel utility evidence with a simple delta rule.
  - `frozen = True` pins gains at their current values -> the FIXED
    salience-rules baseline for K4.
  - The `attention_update` consumer (consumers.py) calls `update_gains`
    from broadcast feedback, closing the learning loop through the
    broadcast path — gains are never touched by a side channel.

Cognitive neutrality: channels are opaque labels; identical observations
on two novel channels produce identical bids (tested in identity symmetry
check). No identity-privileged machinery.
"""
from __future__ import annotations

import math

try:
    from .bids import RunningZScoreBid, NonFiniteBidRefused
except ImportError:  # standalone script run (package dir has a dash)
    from bids import RunningZScoreBid, NonFiniteBidRefused

NO_WINNER = "<none>"


class ArbitrationRefused(ValueError):
    """Fail-closed intake."""


class AttentionArbitrator:
    """Deterministic arbitration over habituated bids with learned gains."""

    def __init__(self, channels, *, alpha=0.01, initial_var=1.0,
                 gain_lr=0.1, frozen=False, gain_cap=2.0):
        channels = list(channels)
        if not channels or len(set(channels)) != len(channels):
            raise ArbitrationRefused("channels must be a non-empty unique list")
        self.channels = channels
        self.gain_lr = float(gain_lr)
        if not 0.0 < self.gain_lr <= 1.0:
            raise ArbitrationRefused("gain_lr must be in (0, 1]")
        if not math.isfinite(gain_cap) or gain_cap <= 1.0:
            raise ArbitrationRefused("gain_cap must be finite and > 1.0")
        self.gain_cap = float(gain_cap)
        self.frozen = bool(frozen)
        self._bids = {c: RunningZScoreBid(alpha=alpha, initial_var=initial_var)
                      for c in channels}
        self.gains = {c: 1.0 for c in channels}     # learned attention weights
        self.cycles = 0
        self.gain_history: list[dict] = []          # learning evidence trail

    # ------------------------------------------------------------------
    # arbitration (the selection step — graded bids, no threshold here)
    # ------------------------------------------------------------------
    def arbitrate(self, stimuli: dict) -> dict:
        if set(stimuli) != set(self.channels):
            raise ArbitrationRefused(
                f"stimuli must carry exactly {self.channels}")
        for c, v in stimuli.items():
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ArbitrationRefused(
                    f"stimuli[{c!r}] must be finite, got {v!r}")
        raw = {c: self._bids[c].bid(float(stimuli[c])) for c in self.channels}
        competed = {c: raw[c] * self.gains[c] for c in self.channels}
        ranked = sorted(competed.items(),
                        key=lambda kv: (-kv[1], self.channels.index(kv[0])))
        winner, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        self.cycles += 1
        return {"winner": winner, "margin": top - runner,
                "raw_bids": raw, "competed_bids": competed,
                "gains": dict(self.gains), "cycle": self.cycles}

    # ------------------------------------------------------------------
    # learned gains (K4) — the ONLY legitimate gain writer
    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Delta-rule gain update from per-channel utility evidence.

        ONLY channels present in ``utility`` are updated — a channel with
        no observed utility keeps its gain (unobserved arms are not
        punished; punishing them collapses exploration and freezes the
        policy on the first winner, which we measured). For the updated
        channel: gain += lr * (utility - baseline) * gain, floored at 0.01.

        Returns the new gains. When frozen, returns unchanged (the fixed
        baseline condition). Records (cycle, old, new) in gain_history —
        the evidence that learning actually happened (§25).
        """
        if self.frozen:
            return dict(self.gains)
        old = dict(self.gains)
        for c in self.channels:
            if c not in utility:
                continue  # no evidence -> no update (honest credit assignment)
            u = float(utility[c])
            self.gains[c] = min(self.gain_cap,
                                max(0.01, self.gains[c]
                                    + self.gain_lr * (u - reward_baseline)))
        self.gain_history.append({"cycle": self.cycles,
                                  "before": old, "after": dict(self.gains)})
        return dict(self.gains)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
