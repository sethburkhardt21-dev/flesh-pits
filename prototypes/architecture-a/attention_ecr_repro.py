"""Architecture A -- Return-Conditioned Episodic gain rule (independent replication).

INDEPENDENT REIMPLEMENTATION of the ECR specification for EXP-FP-0080
(replication of EXP-FP-0021). Written from the preregistered rule summary
alone; shares no code with attention_ecr.py beyond the common
AttentionArbitrator base class. attention.py is UNTOUCHED.

RULE (return-conditioned, baseline-free, state-blind)
----------------------------------------------------
Per tick the rule sees ONLY the (arbitration winner, reward) stream.
Episode boundaries are inferred from that stream alone:

* reward >= 1.0            -> the episode just ended in terminal success.
* ticks_this_episode >= max_steps without a success -> the episode must
  have truncated (episodes are at most max_steps ticks), so a new one
  began.

Update paths:

* SUCCESS (r >= 1.0): the episode's critical decision -- its FIRST
  branch-channel winner -- and its progress channel -- the channel with
  the most shaping receipts THIS episode (reliably the forward-motion
  channel, since only forward moves earn shaping) -- are SET to the gain
  cap. A set (not an increment) keeps the two tied so neither can
  dominate and poison the t=0 branch decision (the K9 failure mode).
  The episode ledger is then cleared.

* SHAPING (0 < r < 1.0): a corridor progress tick. Count it toward the
  progress-channel identification; demote every channel that is neither
  the winner nor a branch channel by `shape_punish` (branches are
  enablers, never punished); boost the winner by `corridor_boost`. The
  corridor boost is safe because the pre-branch demotion below caps the
  winner's pre-branch dominance -- corridor strength does not leak into
  the t=0 decision, giving phase-dependent behavior from one gain vector.

* PRE-BRANCH NO-OP (r == 0 while no branch channel has won yet this
  episode, winner not a branch channel): the winner wasted the decision.
  Demote it by `prebranch_demote`, but ONLY down to the current maximum
  branch-channel gain -- never below it. This is the negative-feedback
  loop against t=0 poisoning; demoting below the branches would invert
  the problem into branch dominance.

* r == 0 otherwise: no gain change.

DOCUMENTED PRIORS (given, not learned -- stated honestly as in the
original):
* branch_channels: which channels count as branch decisions. The bandit
  cannot learn this from per-tick rewards (a branch choice earns no
  direct reward).
* max_steps: episode length bound, used ONLY for truncation detection.

Stdlib only. Deterministic given the (winner, reward) sequence.
"""
from __future__ import annotations

try:
    from .attention import AttentionArbitrator
except ImportError:  # run as a standalone script from the package dir
    from attention import AttentionArbitrator


_GAIN_FLOOR = 0.01
_SUCCESS_REWARD = 1.0  # r >= 1.0 is the terminal-success signal


class _EpisodeLedger:
    """Bookkeeping for the current episode, inferred from rewards alone.

    The agent never observes the environment; the ledger reconstructs
    episode structure from the (winner, reward) stream.
    """

    def __init__(self, channels):
        self.ticks = 0
        self.winners = []                    # arbitration winners, in order
        self.shaping_receipts = {c: 0 for c in channels}

    def record(self, winner):
        self.ticks += 1
        if winner is not None:
            self.winners.append(winner)

    def note_shaping(self, winner):
        if winner is not None:
            self.shaping_receipts[winner] += 1

    def saw_branch(self, branch_channels):
        return any(w in branch_channels for w in self.winners)

    def first_branch(self, branch_channels):
        for w in self.winners:
            if w in branch_channels:
                return w
        return None

    def progress_channel(self, channels):
        """Channel with the most shaping receipts this episode.

        Deterministic: ties break toward canonical channel order. A
        +1.0 success on this task requires >= LENGTH forward moves, each
        earning shaping, so the true progress channel strictly leads.
        """
        best, best_n = None, -1
        for c in channels:
            n = self.shaping_receipts[c]
            if n > best_n:
                best, best_n = c, n
        return best

    def clear(self):
        self.ticks = 0
        self.winners = []
        self.shaping_receipts = {c: 0 for c in self.shaping_receipts}


class ReturnConditionedEpisodicArbitrator(AttentionArbitrator):
    """Episodic return-conditioned gain update (ECR, independent build)."""

    def __init__(self, channels, *, branch_channels=None, max_steps=None,
                 shape_punish=0.02, prebranch_demote=0.05,
                 corridor_boost=0.01, **kwargs):
        # branch_channels / max_steps are structural priors wired
        # driver-side. They are REQUIRED; update_gains fails closed if
        # they were never configured (never silently degrades).
        branch_channels = (set(branch_channels)
                           if branch_channels is not None else None)
        if branch_channels is not None and (
                not branch_channels or not branch_channels <= set(channels)):
            raise ValueError(
                "branch_channels must be a non-empty subset of channels")
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps <= 0:
                raise ValueError("max_steps must be positive")
        for name, value in (("shape_punish", shape_punish),
                            ("prebranch_demote", prebranch_demote),
                            ("corridor_boost", corridor_boost)):
            v = float(value)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        super().__init__(channels, **kwargs)
        self.branch_channels = branch_channels
        self.max_steps = max_steps
        self.shape_punish = float(shape_punish)
        self.prebranch_demote = float(prebranch_demote)
        self.corridor_boost = float(corridor_boost)
        self._ledger = _EpisodeLedger(self.channels)
        self._pending_winner = None

    # ------------------------------------------------------------------
    def arbitrate(self, stimuli: dict) -> dict:
        decision = super().arbitrate(stimuli)
        self._pending_winner = decision["winner"]
        return decision

    # ------------------------------------------------------------------
    def _clamp_gains(self):
        for c in self.channels:
            self.gains[c] = min(self.gain_cap, max(_GAIN_FLOOR,
                                                   self.gains[c]))

    def _absorb_success(self):
        """Lock in the episode's branch choice + progress channel."""
        branch_choice = self._ledger.first_branch(self.branch_channels)
        progress = self._ledger.progress_channel(self.channels)
        note = {}
        if branch_choice is not None:
            self.gains[branch_choice] = self.gain_cap
            note["branch_credit"] = branch_choice
        else:
            # A +1.0 without any branch win is structurally impossible
            # on this task; record rather than crash.
            note["branch_credit"] = None
        if progress is not None:
            self.gains[progress] = self.gain_cap
            note["progress_credit"] = progress
        self._ledger.clear()
        return note

    def _absorb_shaping(self, winner):
        """Corridor progress tick: identify, punish the useless, boost."""
        self._ledger.note_shaping(winner)
        note = {}
        if self.corridor_boost > 0.0:
            self.gains[winner] = min(self.gain_cap,
                                     self.gains[winner] + self.corridor_boost)
            note["corridor_boost"] = winner
        if self.shape_punish > 0.0:
            punished = [c for c in self.channels
                        if c != winner and c not in self.branch_channels]
            for c in punished:
                self.gains[c] = max(_GAIN_FLOOR,
                                    self.gains[c] - self.shape_punish)
            if punished:
                note["punished"] = punished
        return note

    def _absorb_prebranch_noop(self, winner):
        """Wasted pre-branch decision: negative feedback, floored."""
        branch_ceiling = max(self.gains[c] for c in self.branch_channels)
        if self.gains[winner] > branch_ceiling:
            self.gains[winner] = max(branch_ceiling,
                                     self.gains[winner]
                                     - self.prebranch_demote)
            return {"prebranch_demote": winner}
        return {}

    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Episodic return-conditioned gain update.

        ``reward_baseline`` is accepted for interface compatibility and
        ignored: the rule is baseline-free by design.
        """
        if self.frozen:
            return dict(self.gains)
        if self.branch_channels is None or self.max_steps is None:
            raise ValueError(
                "ReturnConditionedEpisodicArbitrator requires "
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
        detail = {"reward": reward, "winner": winner}

        if reward >= _SUCCESS_REWARD:
            detail.update(self._absorb_success())
        elif reward > 0.0:
            if winner is not None:
                detail.update(self._absorb_shaping(winner))
        elif reward == 0.0 and pre_branch and winner is not None \
                and winner not in self.branch_channels:
            detail.update(self._absorb_prebranch_noop(winner))
        # r == 0 post-branch (or winner-less): no gain change.

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
