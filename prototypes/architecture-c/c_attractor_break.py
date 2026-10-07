"""Architecture C — attractor-break intervention variants (EXP-FP-0011).

ADDITIVE ONLY: canonical c_agent.py, attention.py, attention_cue.py,
tick.py are UNTOUCHED (G0). Both interventions subclass the canonical
classes and override exactly one method each.

Mechanism under test (EXP-FP-0010, measured): gains floored at 0.01 on
0.997 of channel-episodes -> arbitration = argmax of habituated raw bids
-> raw bids respond only to stimulus CHANGES (RunningZScoreBid) -> a
wall-bumping loop yields static obs -> static stimuli -> frozen winner ->
a self-reinforcing bump attractor (stuck 0.972, return -5.88).

Two independent interventions, preregistered
(experiments/preregistration_ATTRACTOR_BREAK.json):

(a) RepeatPenaltyArbitrator — anti-habituation: penalize recently-repeated
    actions in the competed bid. Targets the self-reinforcing loop
    directly. PENALTY=0.02, WINDOW=4 (scale from 0010 frozen margins
    0.001-0.006).
(b) LiftedFloorArbitrator — gain-floor lift: update_gains floors at
    gain_floor=0.5 instead of 0.01, so the predictor's stimulus variation
    can reach action selection. Targets the integration-failure half.

Stdlib only. Deterministic given seeds. No identity machinery: channels
are opaque labels and the penalty is symmetric across channels.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_A = os.path.join(os.path.dirname(_HERE), "architecture-a")
for _p in (_HERE, _ARCH_A):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_agent import ArchC  # noqa: E402
from attention import AttentionArbitrator  # noqa: E402
from attention_cue import CueIndexedArbitrator  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Anti-habituation: repeat penalty on the competed bid.
# ---------------------------------------------------------------------------
class RepeatPenaltyArbitrator(AttentionArbitrator):
    """Subtract PENALTY * (wins by channel c in the last WINDOW ticks)
    from each channel's competed bid, then take the argmax with the
    canonical channel-order tie-break.

    The base class's arbitrate() is called exactly once per tick (bid
    state and cycle advance exactly as in the canonical path); only the
    winner/margin are recomputed from the penalized bids. Everything
    downstream (buffer admission, _select_action) sees the penalized
    competed bids — that is the intervention's causal path.
    """

    PENALTY = 0.02
    WINDOW = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._win_history: list = []  # recent winners, channel labels

    def arbitrate(self, stimuli: dict) -> dict:
        decision = super().arbitrate(stimuli)
        recent = self._win_history[-self.WINDOW:]
        penalty = {c: self.PENALTY * recent.count(c)
                   for c in self.channels}
        penalized = {c: decision["competed_bids"][c] - penalty[c]
                     for c in self.channels}
        ranked = sorted(penalized.items(),
                        key=lambda kv: (-kv[1], self.channels.index(kv[0])))
        winner, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        self._win_history.append(winner)
        out = dict(decision)
        out.update({"winner": winner, "margin": top - runner,
                    "competed_bids": penalized,
                    "repeat_penalty": dict(penalty)})
        return out


class RepeatPenaltyCueArbitrator(RepeatPenaltyArbitrator,
                                CueIndexedArbitrator):
    """Repeat-penalty variant of the cue-indexed arbitrator (control envs).

    MRO: RepeatPenaltyCueArbitrator -> RepeatPenaltyArbitrator ->
    CueIndexedArbitrator -> AttentionArbitrator. Both parents cooperate
    via *args/**kwargs and super() chains; CueIndexedArbitrator's gains
    property dispatches per context as in the canonical path.
    """


# ---------------------------------------------------------------------------
# (b) Gain-floor lift: the gain loop keeps a usable baseline.
# ---------------------------------------------------------------------------
class LiftedFloorArbitrator(AttentionArbitrator):
    """Identical to AttentionArbitrator.update_gains except the floor:
    gains are floored at gain_floor (default 0.5) instead of 0.01.

    With the floor lifted, learned gains stay in [0.5, gain_cap], so the
    predictor's stimulus variation reaches action selection even under
    the sparse punishment that floors the canonical 0.01 loop.
    """

    def __init__(self, *args, gain_floor: float = 0.5, **kwargs):
        if not (0.0 < gain_floor <= 1.0):
            raise ValueError("gain_floor must be in (0, 1]")
        self.gain_floor = float(gain_floor)
        super().__init__(*args, **kwargs)

    def update_gains(self, utility: dict, reward_baseline: float = 0.0
                     ) -> dict:
        """The canonical delta rule, floor lifted. Body mirrors
        AttentionArbitrator.update_gains line-for-line except 0.01 ->
        self.gain_floor; no other behavioral change."""
        if self.frozen:
            return dict(self.gains)
        old = dict(self.gains)
        for c in self.channels:
            if c not in utility:
                continue  # no evidence -> no update (canonical rule)
            u = float(utility[c])
            self.gains[c] = min(self.gain_cap,
                                max(self.gain_floor, self.gains[c]
                                    + self.gain_lr * (u - reward_baseline)))
        self.gain_history.append({"cycle": self.cycles,
                                  "before": old, "after": dict(self.gains)})
        return dict(self.gains)


class LiftedFloorCueArbitrator(LiftedFloorArbitrator, CueIndexedArbitrator):
    """Gain-floor-lift variant of the cue-indexed arbitrator (control
    envs). MRO: LiftedFloorCueArbitrator -> LiftedFloorArbitrator ->
    CueIndexedArbitrator -> AttentionArbitrator; the gains property
    dispatches per context exactly as in the canonical path."""


# ---------------------------------------------------------------------------
# ArchC subclass wiring the variant arbitrators (canonical c_agent.py
# untouched: ArchC.__init__ is re-implemented here with ONLY the
# arbitrator-class selection changed; every other line is identical).
# ---------------------------------------------------------------------------
class ArchCAttractorBreak(ArchC):
    """ArchC with swappable arbitrator classes for the attractor-break
    interventions. Subclasses set PLAIN_ARB_CLS / CUE_ARB_CLS."""

    PLAIN_ARB_CLS = AttentionArbitrator
    CUE_ARB_CLS = CueIndexedArbitrator

    def __init__(self, config: dict):
        # NOTE: line-for-line copy of ArchC.__init__ except the arb_cls
        # selection below (marked VARIANT). Keeps c_agent.py canonical.
        self.config = dict(config)
        channels = list(config["channels"])
        self.channels = channels
        self.channel_to_action = dict(config["channel_to_action"])
        self.observation_space = dict(config["observation_space"])
        self.env_name = config["env_name"]
        from c_agent import (categorical_slices, make_predictor_specialists,
                             HierarchicalGenerativeModel, EpisodicStore,
                             DisabledStore, WorkspaceTick)
        self.cat_slices = categorical_slices(self.observation_space)
        self.kappa = float(config.get("kappa", 0.1))
        seed = int(config.get("agent_seed", 0))

        pk = dict(config.get("predictor_kwargs", {}))
        pk.setdefault("seed", seed)
        self.predictor = HierarchicalGenerativeModel(
            obs_dim=sum(1 if s["type"] == "scalar" else s["shape"]
                        if s["type"] == "vector" else len(s["values"])
                        for s in self.observation_space.values()),
            n_actions=len(channels), cat_slices=self.cat_slices, **pk)
        self.memory = (EpisodicStore() if config.get("memory_enabled", True)
                       else DisabledStore())
        self.frozen_predictor = bool(config.get("frozen_predictor", False))
        if self.frozen_predictor:
            self.predictor.eta0 = 0.0
            self.predictor.etaD = 0.0
            self.predictor.eta_r = 0.0
            self.predictor.etaR = 0.0

        specialists = make_predictor_specialists(
            channels, self.predictor, self.memory, self.observation_space,
            self.cat_slices, kappa=self.kappa)
        context_fn = config.get("context_fn")
        # VARIANT: the only line that differs from ArchC.__init__.
        arb_cls = (self.CUE_ARB_CLS if context_fn is not None
                   else self.PLAIN_ARB_CLS)
        self.tick = WorkspaceTick(
            channels, specialists, capacity=len(channels),
            gain_lr=float(config.get("gain_lr", 0.15)),
            frozen_gains=bool(config.get("frozen_gains", False)),
            ignition_kwargs={"theta": float(config.get("theta", 0.45))},
            arbitrator_cls=arb_cls, context_fn=context_fn)

        self._tick_index = 0
        self._episode_idx = -1
        self._state_vec = None


class ArchCRepeatPenalty(ArchCAttractorBreak):
    """C + anti-habituation repeat penalty."""

    NAME = "arch_c_repeat_penalty"
    PLAIN_ARB_CLS = RepeatPenaltyArbitrator
    CUE_ARB_CLS = RepeatPenaltyCueArbitrator


class ArchCGainLift(ArchCAttractorBreak):
    """C + gain-floor lift (0.5)."""

    NAME = "arch_c_gain_lift"
    PLAIN_ARB_CLS = LiftedFloorArbitrator
    CUE_ARB_CLS = LiftedFloorCueArbitrator
