"""Architecture A — Episodic Contrastive Return (ECR) gain rule.

A NEW gain-update rule for sparse delayed reward. NOT the constant-baseline
delta rule (attention.py). NOT K9's eligibility-trace rule
(attention_sparse.py).

DESIGN
------
Return-conditioned and baseline-free. Uses ONLY the per-tick
(winner, reward) stream -- no observation, no env state (state-blind).

Key mechanism: EPISODIC credit assignment with explicit episode-boundary
detection from the reward stream alone:
  * r >= 1.0  -> terminal success signal: the episode just ended.
  * ep_ticks >= max_steps without r >= 1.0 -> the episode MUST have
    truncated (episodes are <= max_steps ticks), so a new episode began.

On a successful episode (r >= 1.0):
  * bc   = first branch_channels winner in this episode's winner list
           (the branch choice -- the episode's critical decision; a +1.0
           is impossible without one, so bc always exists).
  * prog = argmax of per-channel shaping receipts (the channel that
           earns the 0.02 progress rewards -- reliably "forward", since
           only forward moves earn shaping and a +1.0 requires >=9 of them).
  * gain[bc]   = gain_cap   (lock in the successful branch choice)
  * gain[prog] = gain_cap   (lock in the progress channel)
  No other channel is touched. Because bc and prog are SET (not
  incremented) to the same cap, they stay tied/competitive -- neither
  can dominate and poison the t=0 decision (the K9 failure mode).
  Channels that never participate in success (e.g. stay) are suppressed
  BY RELATIVITY: they sit at 1.0 while bc/prog sit at 2.0.

On a pre-branch tick (no branch channel has won yet this episode,
detected via the episode winner list): if the winner is NOT a branch
channel and r == 0, it was a wasted no-op (forward/stay do nothing before
the branch is chosen). Demote it by prebranch_demote. This is a NEGATIVE
FEEDBACK loop against t=0 poisoning (the K9 failure mode): the more
forward dominates pre-branch, the more it is demoted. Branch channels are
never demoted pre-branch (choosing branch is correct there). Once a branch
is chosen, the demotion stops (we're in the corridor; forward is useful).

On shaping (0 < r < 1.0, always a forward move in the corridor): count it
toward prog identification, demote provably-useless channels by
shape_punish, AND boost the winner (forward) by corridor_boost. The
corridor boost is SAFE because the t=0 demotion caps forward's dominance:
forward can be strong in the corridor (more shaping, more +1.0 hits)
without poisoning t=0. This breaks the stationary-policy limit -- it is
effectively phase-dependent behavior from a persistent gain vector.

On r == 0 (non-t=0): nothing.

DOCUMENTED TASK-STRUCTURAL PRIORS (limitations, stated honestly):
  * branch_channels: which channels are "branch decisions". The bandit
    cannot learn this from per-tick rewards (branch never earns direct
    reward); it is given as structure, like K9's trace_lambda=0.9 was
    tuned to the 10-step credit gap.
  * max_steps: episode length bound, used ONLY for truncation detection.
    Passed explicitly by the driver from the env interface.
  * shape_punish: per-shaping-tick demotion of provably-useless channels.
    Small enough that ~50 shaping ticks floor a channel (no abrupt
    policy swing); large enough to matter within a run.

Stdlib only. Deterministic given the (winner, reward) sequence.
Subclasses AttentionArbitrator; attention.py is UNTOUCHED.
"""
from __future__ import annotations

try:
    from .attention import AttentionArbitrator
except ImportError:  # standalone script run (package dir has a dash)
    from attention import AttentionArbitrator


class EpisodicContrastiveArbitrator(AttentionArbitrator):
    """Episodic contrastive return gain update (ECR)."""

    SUCCESS_THRESHOLD = 1.0  # r >= 1.0 means terminal success (+1.0 goal)

    def __init__(self, channels, *, branch_channels=None, max_steps=None,
                 shape_punish=0.02, prebranch_demote=0.05,
                 corridor_boost=0.01, **kwargs):
        # branch_channels / max_steps are task-structural priors, wired
        # driver-side (post-construction or via tick.py arbitrator_kwargs).
        # They are REQUIRED for the rule to operate; update_gains fails
        # closed if they are not configured (never silently degrades).
        branch_channels = (set(branch_channels) if branch_channels is not None
                           else None)
        if branch_channels is not None and (
                not branch_channels or not branch_channels <= set(channels)):
            raise ValueError(
                "branch_channels must be a non-empty subset of channels")
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps <= 0:
                raise ValueError("max_steps must be positive")
        sp = float(shape_punish)
        if not 0.0 <= sp <= 1.0:
            raise ValueError("shape_punish must be in [0, 1]")
        t0d = float(prebranch_demote)
        if not 0.0 <= t0d <= 1.0:
            raise ValueError("prebranch_demote must be in [0, 1]")
        cb = float(corridor_boost)
        if not 0.0 <= cb <= 1.0:
            raise ValueError("corridor_boost must be in [0, 1]")
        self.branch_channels = branch_channels
        self.max_steps = max_steps
        self.shape_punish = sp
        self.prebranch_demote = t0d
        self.corridor_boost = cb
        self._ep_winners: list = []      # winners this episode, in order
        self._ep_ticks = 0               # ticks since episode start
        self._shaping_counts = {c: 0 for c in list(channels)}
        self._last_winner = None
        super().__init__(channels, **kwargs)

    # ------------------------------------------------------------------
    def arbitrate(self, stimuli: dict) -> dict:
        decision = super().arbitrate(stimuli)
        self._last_winner = decision["winner"]
        return decision

    # ------------------------------------------------------------------
    def _progress_channel(self):
        """Channel with the most shaping receipts (the progress channel).

        Well-defined whenever a success is being processed: a +1.0
        requires >= LENGTH forward moves, each earning shaping, so the
        progress channel strictly leads the counts.
        """
        best, best_n = None, -1
        for c in self.channels:  # canonical order tie-break (deterministic)
            n = self._shaping_counts[c]
            if n > best_n:
                best, best_n = c, n
        return best

    def _reset_episode(self):
        self._ep_winners = []
        self._ep_ticks = 0

    # ------------------------------------------------------------------
    def update_gains(self, utility: dict, reward_baseline: float = 0.0) -> dict:
        """Episodic contrastive return update.

        ``reward_baseline`` is accepted for interface compatibility and
        ignored: this rule is baseline-free by design.
        """
        if self.frozen:
            return dict(self.gains)
        if self.branch_channels is None or self.max_steps is None:
            raise ValueError(
                "EpisodicContrastiveArbitrator requires branch_channels and "
                "max_steps to be configured (driver-side structural priors); "
                "refusing to update gains unconfigured.")
        old = dict(self.gains)
        r = sum(float(v) for v in utility.values())
        w = self._last_winner
        info = {"return": r, "winner": w}

        # Pre-branch detection BEFORE appending: no branch channel has won
        # yet this episode means we are still pre-branch (t=0-ish); a
        # non-branch winner now is a wasted no-op.
        pre_branch = not any(c in self.branch_channels
                             for c in self._ep_winners)

        # Every tick belongs to the current episode until a boundary.
        if w is not None:
            self._ep_winners.append(w)
        self._ep_ticks += 1

        if r >= self.SUCCESS_THRESHOLD:
            # ---- successful episode: lock in branch choice + progress ----
            bc = next((c for c in self._ep_winners
                       if c in self.branch_channels), None)
            prog = self._progress_channel()
            if bc is not None:
                self.gains[bc] = self.gain_cap
                info["branch_credit"] = bc
            else:
                # Fail-safe: a +1.0 without a branch win should be
                # impossible; record and skip rather than crash.
                info["branch_credit"] = None
            if prog is not None:
                self.gains[prog] = self.gain_cap
                info["progress_credit"] = prog
            self._reset_episode()
        elif r > 0.0:
            # ---- shaping (corridor forward move): identify progress,
            # demote the provably-useless, boost the corridor winner ----
            if w is not None:
                self._shaping_counts[w] += 1
                if self.corridor_boost > 0.0:
                    self.gains[w] = min(self.gain_cap,
                                        self.gains[w] + self.corridor_boost)
                    info["corridor_boost"] = w
                if self.shape_punish > 0.0:
                    demote = [c for c in self.channels
                              if c != w and c not in self.branch_channels]
                    for c in demote:
                        self.gains[c] = max(0.01,
                                            self.gains[c] - self.shape_punish)
                    if demote:
                        info["demoted"] = demote
        elif r == 0.0 and pre_branch and w is not None \
                and w not in self.branch_channels:
            # ---- pre-branch no-op: a non-branch winner before any branch
            # was chosen wasted the decision. Demote (negative feedback
            # vs t=0 poisoning) -- BUT only if it is currently ABOVE the
            # branch gains. Demoting below the branches would invert the
            # problem (branch no-ops crowding out corridor forwards).
            # Branch winners pre-branch are correct: untouched.
            branch_max = max(self.gains[c] for c in self.branch_channels)
            if self.gains[w] > branch_max:
                self.gains[w] = max(branch_max,
                                    self.gains[w] - self.prebranch_demote)
                info["prebranch_demoted"] = w
        # r == 0 (post-branch): no gain change.

        # Truncation detection: no success within max_steps ticks means
        # the episode ended and a new one began.
        if self._ep_ticks >= self.max_steps:
            info["truncated_reset"] = True
            self._reset_episode()

        # Enforce bounds (cap/floor) in case of direct sets.
        for c in self.channels:
            self.gains[c] = min(self.gain_cap, max(0.01, self.gains[c]))

        self.gain_history.append({"cycle": self.cycles,
                                  "before": old, "after": dict(self.gains),
                                  **info})
        return dict(self.gains)
