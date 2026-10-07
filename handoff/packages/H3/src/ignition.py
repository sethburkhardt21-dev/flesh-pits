"""Architecture A — recurrent ignition with nonlinear gating.

Mechanism: the selected content's salience x drives a recurrent
amplification loop to convergence (or a hard cycle cap):

    s_{n+1} = sigmoid( gain * (x_eff - theta) + feedback * (s_n - 0.5) )

with feedback > 0, the loop is BISTABLE: it settles at ~1 (ignited) or
~0 (quenched), never graded. Propagation to broadcast is then gated by a
HARD Heaviside step on the settled state:

    ignited = (s_final >= 0.5)

K3 requirement: sub-threshold content must NOT propagate to consumers
(no graded leak); supra-threshold content must, with a sharp transition.
The linear/graded control (``linear_probe``) replaces the gate with
``strength_out = x`` to test the kill condition honestly.

``x_eff = x + jitter``: per-trial seeded jitter lets the K3 probe
measure propagation PROBABILITY over salience (a deterministic unit
cannot produce a probability curve otherwise).

Reverberation: ``strength`` persists across ticks (s_0 of the next
ignition = current strength) — working-memory persistence, not a claim
about anything more.
"""
from __future__ import annotations

import math

try:
    from .bids import stable_sigmoid
except ImportError:  # standalone script run
    from bids import stable_sigmoid


class IgnitionRefused(ValueError):
    pass


class RecurrentIgnition:
    def __init__(self, *, theta=0.6, gain=12.0, feedback=4.0,
                 max_cycles=50, eps=1e-6, linear_probe=False):
        """linear_probe=True replaces the bistable gate with graded
        passthrough (the K3 kill-control: if the step vanishes under the
        REAL gate but not under this probe, the probe was measuring the
        sigmoid of the bid function, not an ignition event)."""
        for name, v in (("theta", theta), ("gain", gain), ("feedback", feedback)):
            if not math.isfinite(v):
                raise IgnitionRefused(f"{name} must be finite")
        if not 0.0 < theta < 1.0:
            raise IgnitionRefused("theta must be in (0, 1)")
        if gain <= 0 or feedback < 0:
            raise IgnitionRefused("gain > 0, feedback >= 0 required")
        self.theta = float(theta)
        self.gain = float(gain)
        self.feedback = float(feedback)
        self.max_cycles = int(max_cycles)
        self.eps = float(eps)
        self.linear_probe = bool(linear_probe)
        self.strength = 0.0          # reverberant state, persists across ticks
        self.last_ignited = False
        self.last_cycles = 0
        self.history: list[dict] = []

    def ignite(self, salience: float, jitter: float = 0.0) -> dict:
        x = float(salience)
        if not math.isfinite(x):
            raise IgnitionRefused("salience must be finite")
        x_eff = x + float(jitter)

        if self.linear_probe:
            # kill-control: graded passthrough, NO gate, NO recurrence
            self.strength = x_eff
            self.last_ignited = True  # everything "propagates" graded
            self.last_cycles = 1
            rec = {"salience": x, "jitter": float(jitter),
                   "strength": x_eff, "ignited": True,
                   "cycles": 1, "converged": True, "mode": "linear_probe"}
            self.history.append(rec)
            return rec

        s = self.strength  # reverberant seed from the previous tick
        cycles, converged = 0, False
        for n in range(self.max_cycles):
            cycles = n + 1
            drive = self.gain * (x_eff - self.theta) + self.feedback * (s - 0.5)
            s_next = stable_sigmoid(drive)
            if abs(s_next - s) < self.eps:
                s = s_next
                converged = True
                break
            s = s_next
        self.strength = s
        ignited = bool(s >= 0.5)     # THE GATE: hard Heaviside, no leak
        self.last_ignited = ignited
        self.last_cycles = cycles
        rec = {"salience": x, "jitter": float(jitter), "x_eff": x_eff,
               "strength": s, "ignited": ignited,
               "cycles": cycles, "converged": converged, "mode": "bistable_gate"}
        self.history.append(rec)
        return rec

    def reset_strength(self):
        self.strength = 0.0
