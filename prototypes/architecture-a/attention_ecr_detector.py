"""EXP-FP-0126 -- adaptive terminal-success detector for ECR (additive).

EXP-FP-0120 proved the hardcoded `reward >= 1.0` terminal detector in
ReturnConditionedEpisodicArbitrator silently misfires when same-tick
penalties pull the terminal reward below 1.0 (grid_world goal tick =
-0.02 + 1.00 = 0.98): the goal tick is absorbed as a shaping tick and
the episodic lock-in never executes.

This module subclasses ReturnConditionedEpisodicArbitrator
(attention_ecr_repro.py, UNTOUCHED) and overrides ONLY the terminal
classification in update_gains, via the _is_terminal(reward) hook. All
other paths -- shaping absorption, pre-branch demotion, r==0,
truncation detection, clamping, gain_history -- are inherited
behaviorally identical (the method body is a verbatim copy of the
parent's with the single classification site replaced). The agent stays
state-blind: the detector reads the reward stream alone.

Detector variants (constructor kwarg detector=):
  "hardcoded" : r >= 1.0  -- current ECR; the 0120 failure mode (D0).
  "floor"     : r >= FLOOR  -- FLOOR=0.1 (D1).
  "baseline"  : r > max(K*B_pos, FLOOR), K=2.0, B_pos = EMA(alpha=0.1)
                of r>0 ticks (D2).
  "relmax"    : r >= max(FLOOR, REL*r_big), REL=0.5, r_big = running max
                of r>0 ticks (D3, the H1 detector).

Statistics update on every r>0 tick AFTER classification (novel relative
to history); terminal ticks update the stats too. The floor 0.1 is a
documented scale prior (one order of magnitude below terminal-class
rewards, above per-tick shaping/penalties <= ~0.05) -- not learned.

Stdlib only. Deterministic given the (winner, reward) sequence.
"""
from __future__ import annotations

try:
    from .attention_ecr_repro import (
        ReturnConditionedEpisodicArbitrator,
        _GAIN_FLOOR,
        _SUCCESS_REWARD,
        _EpisodeLedger,
    )
except ImportError:  # run as a standalone script from the package dir
    from attention_ecr_repro import (
        ReturnConditionedEpisodicArbitrator,
        _GAIN_FLOOR,
        _SUCCESS_REWARD,
        _EpisodeLedger,
    )


_FLOOR = 0.1   # scale floor: terminal-class rewards are >= 10x per-tick noise
_REL = 0.5     # relmax: terminal must reach >= half of the best positive seen
_K = 2.0       # baseline: terminal must exceed 2x the positive-reward mean
_ALPHA = 0.1   # EMA rate for the baseline detector's B_pos

_DETECTORS = ("hardcoded", "floor", "baseline", "relmax")


class AdaptiveTerminalECR(ReturnConditionedEpisodicArbitrator):
    """ECR with a scale-relative terminal-success detector."""

    def __init__(self, channels, *, detector="relmax", progress_guard=False,
                 **kwargs):
        if detector not in _DETECTORS:
            raise ValueError(
                f"detector must be one of {_DETECTORS}, got {detector!r}")
        super().__init__(channels, **kwargs)
        self.detector = detector
        # EXP-FP-0127: when True, a terminal tick in an episode with ZERO
        # shaping receipts skips the progress cap-set (the canonical
        # tie-break would pin an arbitrary channel) and locks in the
        # branch choice only. Guard=False is behaviorally identical to
        # the parent.
        self.progress_guard = bool(progress_guard)
        # Detector statistics: reward-stream only (state-blind preserved).
        self._r_big = 0.0     # running max of r>0 ticks (relmax)
        self._b_pos = 0.0     # EMA of r>0 ticks (baseline)
        self._n_pos = 0       # count of r>0 ticks

    # ------------------------------------------------------------------
    def _absorb_success(self):
        """Lock in the episode's branch choice (+ progress, unless guarded).

        Parent-identical when progress_guard=False. When the guard is on
        and the episode contains zero shaping receipts, the progress
        channel is unidentifiable from the reward stream, so the
        canonical tie-break (which would pin an arbitrary channel at the
        gain cap) is disarmed: branch choice is locked in alone.
        """
        branch_choice = self._ledger.first_branch(self.branch_channels)
        note = {}
        if branch_choice is not None:
            self.gains[branch_choice] = self.gain_cap
            note["branch_credit"] = branch_choice
        else:
            note["branch_credit"] = None
        shaping_total = sum(self._ledger.shaping_receipts.values())
        if self.progress_guard and shaping_total == 0:
            note["progress_credit"] = None
            note["progress_guarded"] = True
        else:
            progress = self._ledger.progress_channel(self.channels)
            if progress is not None:
                self.gains[progress] = self.gain_cap
                note["progress_credit"] = progress
        self._ledger.clear()
        return note

    # ------------------------------------------------------------------
    def _is_terminal(self, reward: float) -> bool:
        """Classify an r>0 tick as terminal success or shaping.

        Called exactly where the parent's `reward >= _SUCCESS_REWARD`
        sits. reward > 0 is guaranteed by the call site.
        """
        if self.detector == "hardcoded":
            return reward >= _SUCCESS_REWARD
        if self.detector == "floor":
            return reward >= _FLOOR
        if self.detector == "baseline":
            return reward > max(_K * self._b_pos, _FLOOR)
        # relmax
        return reward >= max(_FLOOR, _REL * self._r_big)

    def _update_detector_stats(self, reward: float) -> None:
        """Update running statistics from an r>0 tick, post-classification."""
        if reward > 0.0:
            self._n_pos += 1
            if reward > self._r_big:
                self._r_big = reward
            self._b_pos = ((1.0 - _ALPHA) * self._b_pos
                           + _ALPHA * reward)

    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Episodic return-conditioned gain update, adaptive detector.

        Verbatim copy of the parent's method body with the single
        classification site replaced by _is_terminal(reward) and the
        detector statistics update added. `reward_baseline` is accepted
        for interface compatibility and ignored (baseline-free, as in
        the parent).
        """
        if self.frozen:
            return dict(self.gains)
        if self.branch_channels is None or self.max_steps is None:
            raise ValueError(
                "AdaptiveTerminalECR requires "
                "branch_channels and max_steps (driver-side structural "
                "priors); refusing to update gains unconfigured.")
        gains_before = dict(self.gains)
        reward = sum(float(v) for v in utility.values())
        winner = self._pending_winner

        # Pre-branch status BEFORE recording this tick: if no branch
        # channel has won yet this episode, a non-branch winner now is a
        # wasted no-op at (or near) t=0.
        pre_branch = not self._ledger.saw_branch(self.branch_channels)

        self._ledger.record(winner)
        detail = {"reward": reward, "winner": winner,
                  "detector": self.detector}

        terminal = reward > 0.0 and self._is_terminal(reward)
        detail["terminal"] = bool(terminal)
        if terminal:
            detail.update(self._absorb_success())
        elif reward > 0.0:
            if winner is not None:
                detail.update(self._absorb_shaping(winner))
        elif reward == 0.0 and pre_branch and winner is not None \
                and winner not in self.branch_channels:
            detail.update(self._absorb_prebranch_noop(winner))
        # r == 0 post-branch (or winner-less): no gain change.

        # Detector statistics: after classification, on every r>0 tick.
        self._update_detector_stats(reward)
        detail["r_big"] = self._r_big
        detail["b_pos"] = self._b_pos

        # Truncation detection: max_steps ticks with no terminal success
        # means the episode ended and a new one began.
        if self._ledger.ticks >= self.max_steps:
            detail["truncated_reset"] = True
            self._ledger.clear()

        self._clamp_gains()
        self.gain_history.append({"cycle": self.cycles,
                                  "gains_before": gains_before,
                                  "gains_after": dict(self.gains),
                                  **detail})
        return dict(self.gains)
