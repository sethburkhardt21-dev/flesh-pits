"""Architecture A — sparse-reward gain redesign (K9, NR-A-006 candidate).

Subclasses AttentionArbitrator (attention.py is UNTOUCHED). Replaces the
constant-baseline delta rule with a return-conditioned, baseline-free
eligibility-trace gain update:

  per-tick (in arbitrate):  e_c <- lambda * e_c for all channels,
                            then e_winner += 1   (accumulating traces)
  on feedback (in update_gains): r = sum(utility.values())
                            if r != 0: gain[c] += lr * r * e_c
                                       for all c with e_c > 0
                            if r == 0: NO gain change (traces still decay)

The reward_baseline argument is accepted but DELIBERATELY UNUSED —
removing the baseline IS the redesign (the NR-A-006 decay mechanism was
the constant 0.5 baseline punishing every arm under sparse reward).
Consumer/tick wiring is unchanged: the feedback envelope still carries
``utility`` and ``reward_baseline``; the sole-path discipline holds.

Departure from the base class, documented: the base updates ONLY
channels present in ``utility`` (no evidence -> no update). Here the
eligibility trace IS the evidence record, so credit flows to past
actions with e_c > 0 — that temporal back-propagation is the point.

Stdlib only. Deterministic given the trace/utility sequence.
"""
from __future__ import annotations

try:
    from .attention import AttentionArbitrator
except ImportError:  # standalone script run (package dir has a dash)
    from attention import AttentionArbitrator


class SparseRewardArbitrator(AttentionArbitrator):
    """Return-conditioned, baseline-free gain update via eligibility traces."""

    def __init__(self, channels, *, trace_lambda=0.9, **kwargs):
        lam = float(trace_lambda)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("trace_lambda must be in [0, 1]")
        self.trace_lambda = lam
        self.traces = {c: 0.0 for c in list(channels)}
        super().__init__(channels, **kwargs)

    # ------------------------------------------------------------------
    # arbitration: identical selection; traces ride along
    # ------------------------------------------------------------------
    def arbitrate(self, stimuli: dict) -> dict:
        lam = self.trace_lambda
        for c in self.channels:
            self.traces[c] *= lam
        decision = super().arbitrate(stimuli)
        self.traces[decision["winner"]] += 1.0
        decision["traces"] = dict(self.traces)
        return decision

    # ------------------------------------------------------------------
    # gain update: return-conditioned, baseline-free (K9 redesign)
    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Distribute the return over the eligibility trace.

        ``reward_baseline`` is accepted for interface compatibility with
        the attention_update consumer and then ignored: this update is
        baseline-free by design. Only non-zero return moves gains, so
        sparse reward can no longer decay every arm toward the floor.
        """
        if self.frozen:
            return dict(self.gains)
        old = dict(self.gains)
        r = sum(float(v) for v in utility.values())
        if r != 0.0:
            for c in self.channels:
                e = self.traces.get(c, 0.0)
                if e > 0.0:
                    self.gains[c] = min(
                        self.gain_cap,
                        max(0.01, self.gains[c] + self.gain_lr * r * e))
        # r == 0: no gain change. The trace already decayed in arbitrate().
        self.gain_history.append({"cycle": self.cycles,
                                  "before": old, "after": dict(self.gains),
                                  "return": r,
                                  "traces": dict(self.traces)})
        return dict(self.gains)
