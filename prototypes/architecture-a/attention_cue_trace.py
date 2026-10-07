"""Architecture A — cue-indexed eligibility-trace adapter (K11).

COMPOSITION of the two parent mechanisms (both parent modules UNTOUCHED):
  K10 (attention_cue.CueIndexedArbitrator): one gain vector per context.
  K9  (attention_sparse.SparseRewardArbitrator rule): per-tick eligibility
       traces with a return-conditioned, BASELINE-FREE gain update.

CueTraceArbitrator subclasses CueIndexedArbitrator and adds K9-style
traces, now PER-CONTEXT (traces are evidence tied to the context that
generated them):

  per-tick (in arbitrate):  e_ctx,c <- lambda * e_ctx,c for the ACTIVE
                            context, then e_ctx,winner += 1
  on feedback (in update_gains): r = sum(utility.values())
                            if r != 0: gain_ctx[c] += lr * r * e_ctx[c]
                                       for all c with e_ctx[c] > 0
                            if r == 0: NO gain change (traces still decay)

The reward_baseline argument is accepted but DELIBERATELY UNUSED —
inheriting K9's baseline-free design. Consumer/tick wiring is unchanged.

With a constant context this is the K9 code path (the P2 equivalence
control depends on this); with a cue context on dense reward it is the
trace rule operating on per-cue vectors (P3).

Cognitive neutrality: contexts are opaque hashables; no
identity-privileged machinery.

Stdlib only. Deterministic given the trace/utility/context sequence.
"""
from __future__ import annotations

try:
    from .attention_cue import CueIndexedArbitrator
except ImportError:  # standalone script run (package dir has a dash)
    from attention_cue import CueIndexedArbitrator


class CueTraceArbitrator(CueIndexedArbitrator):
    """Per-cue gain vectors (K10) x per-cue eligibility traces (K9 rule)."""

    def __init__(self, channels, *, trace_lambda=0.9, **kwargs):
        lam = float(trace_lambda)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("trace_lambda must be in [0, 1]")
        self.trace_lambda = lam
        self._context_traces: dict = {}
        super().__init__(channels, **kwargs)

    # ------------------------------------------------------------------
    # per-context trace state (parallel to per-context gain vectors)
    # ------------------------------------------------------------------
    def _traces_for(self, context) -> dict:
        vec = self._context_traces.get(context)
        if vec is None:
            vec = {c: 0.0 for c in self.channels}
            self._context_traces[context] = vec
        return vec

    def all_context_traces(self) -> dict:
        """Inspectable: every context's trace vector (evidence, not control)."""
        return {k: dict(v) for k, v in self._context_traces.items()}

    # ------------------------------------------------------------------
    # arbitration: traces ride along, per active context
    # ------------------------------------------------------------------
    def arbitrate(self, stimuli: dict) -> dict:
        lam = self.trace_lambda
        traces = self._traces_for(self._context)
        for c in self.channels:
            traces[c] *= lam
        decision = super().arbitrate(stimuli)
        traces[decision["winner"]] += 1.0
        decision["traces"] = self.all_context_traces()
        decision["context"] = self._context
        return decision

    # ------------------------------------------------------------------
    # gain update: K9's return-conditioned baseline-free rule, applied to
    # the ACTIVE context's gain vector over the ACTIVE context's traces.
    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Distribute the return over the active context's eligibility trace.

        ``reward_baseline`` is accepted for interface compatibility with
        the attention_update consumer and then ignored: this update is
        baseline-free by design (K9). Only non-zero return moves gains,
        so sparse reward can no longer decay every arm toward the floor.
        """
        if self.frozen:
            return dict(self.gains)
        old = dict(self.gains)
        r = sum(float(v) for v in utility.values())
        if r != 0.0:
            traces = self._traces_for(self._context)
            vec = self.gains  # the ACTIVE context's live gain vector
            for c in self.channels:
                e = traces.get(c, 0.0)
                if e > 0.0:
                    vec[c] = min(
                        self.gain_cap,
                        max(0.01, vec[c] + self.gain_lr * r * e))
        # r == 0: no gain change. The trace already decayed in arbitrate().
        self.gain_history.append({"cycle": self.cycles,
                                  "context": self._context,
                                  "before": old, "after": dict(self.gains),
                                  "return": r,
                                  "traces": dict(
                                      self._traces_for(self._context))})
        return dict(self.gains)
