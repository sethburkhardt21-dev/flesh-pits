"""Architecture A — cue-indexed gain adapter (K10, NR-A-007 candidate).

Subclasses AttentionArbitrator (attention.py is UNTOUCHED). Makes gains
context-conditioned: one gain vector per context value, selected by
``set_context()`` before each arbitration.

  gains property -> the CURRENT context's gain vector (dict channel->gain),
                   lazily created at 1.0 per unseen context.
  set_context(ctx) -> switch the active context (ctx must be hashable).

The update rule is the ORIGINAL constant-baseline delta rule, unchanged:
only the currently active context's vector is updated by update_gains().
With a constant context this module behaves EXACTLY like the base class
(the K10 control probe) — any gap there would be a confound, not a win.

Wiring: the tick calls ``arbitrator.set_context(context_fn(observation))``
before arbitration when a context_fn is provided (tick.py extension;
default None preserves proven behavior).

Cognitive neutrality: contexts are opaque hashables (cue ints); no
identity-privileged machinery.

Stdlib only. Deterministic.
"""
from __future__ import annotations

try:
    from .attention import AttentionArbitrator
except ImportError:  # standalone script run (package dir has a dash)
    from attention import AttentionArbitrator


class CueIndexedArbitrator(AttentionArbitrator):
    """Context-conditioned gains: one learned vector per context value."""

    def __init__(self, channels, **kwargs):
        # Set BEFORE super().__init__: the base __init__ assigns
        # self.gains, which routes through the property setter below.
        self._context = None
        self._context_gains: dict = {}
        super().__init__(channels, **kwargs)

    # ------------------------------------------------------------------
    # context selection
    # ------------------------------------------------------------------
    def set_context(self, context) -> None:
        """Select the active context. Must be hashable (cue ints, None)."""
        try:
            hash(context)
        except TypeError:
            raise ValueError(f"context must be hashable, got {context!r}")
        self._context = context

    @property
    def current_context(self):
        return self._context

    def _gains_for(self, context) -> dict:
        vec = self._context_gains.get(context)
        if vec is None:
            vec = {c: 1.0 for c in self.channels}
            self._context_gains[context] = vec
        return vec

    # ------------------------------------------------------------------
    # gains property: dispatches to the active context's vector
    # ------------------------------------------------------------------
    @property
    def gains(self) -> dict:
        return self._gains_for(self._context)

    @gains.setter
    def gains(self, value: dict) -> None:
        self._context_gains[self._context] = dict(value)

    def all_context_gains(self) -> dict:
        """Inspectable: every context's gain vector (evidence, not control)."""
        return {k: dict(v) for k, v in self._context_gains.items()}
