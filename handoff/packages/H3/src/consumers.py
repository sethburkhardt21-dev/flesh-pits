"""Architecture A — consumer implementations.

Each consumer receives content ONLY via the BroadcastBus (its ``consume``
is the registered handler; no other call path exists in the tick).
Every consumer keeps inspectable state so that "consumption" is a
MEASURABLE effect, not a declaration (K2 verification).

Consumers (all in CONSUMER_REGISTRY):
  memory_admit         -- bounded episodic store; effect: store growth
  self_model_update    -- self-state vector (counts, EMAs); effect: state changes
  planner_input        -- action-proposal queue; effect: queue length / pop
  attention_update     -- learned attention gains from feedback; effect: gains move
  consolidation_eligible -- eligibility marks; effect: mark set growth
  report               -- report lines; effect: lines accumulate
"""
from __future__ import annotations


class Consumer:
    """Base: counts every envelope it actually processes."""

    name = "base"

    def __init__(self):
        self.consumed = 0
        self.last_item_id = None

    def consume(self, consumer: str, envelope: dict) -> None:
        self.consumed += 1
        self.last_item_id = envelope["item_id"]
        self._process(envelope)

    def _process(self, envelope: dict) -> None:
        raise NotImplementedError


class MemoryAdmissionConsumer(Consumer):
    """Episodic memory admission: bounded store of (item_id, kind, summary)."""

    name = "memory_admit"

    def __init__(self, capacity=64):
        super().__init__()
        self.capacity = capacity
        self.store: list[dict] = []

    def _process(self, envelope):
        p = envelope["payload"]
        self.store.append({"item_id": envelope["item_id"],
                           "kind": envelope["kind"],
                           "summary": p.get("summary", ""),
                           "tick": envelope["tick"]})
        if len(self.store) > self.capacity:
            self.store.pop(0)


class SelfModelUpdateConsumer(Consumer):
    """Self-state vector: broadcast count EMA, attended channel, ignition EMA."""

    name = "self_model_update"

    def __init__(self):
        super().__init__()
        self.state = {"broadcast_count": 0, "ignition_rate_ema": 0.0,
                      "last_channel": None}

    def _process(self, envelope):
        p = envelope["payload"]
        self.state["broadcast_count"] += 1
        alpha = 0.1
        self.state["ignition_rate_ema"] = (
            (1 - alpha) * self.state["ignition_rate_ema"]
            + alpha * (1.0 if p.get("ignited") else 0.0))
        if p.get("channel"):
            self.state["last_channel"] = p["channel"]


class PlannerInputConsumer(Consumer):
    """Planning queue: turns workspace items into action proposals."""

    name = "planner_input"

    def __init__(self):
        super().__init__()
        self.queue: list[dict] = []

    def _process(self, envelope):
        p = envelope["payload"]
        self.queue.append({"item_id": envelope["item_id"],
                           "proposed_action": p.get("proposed_action"),
                           "channel": p.get("channel"),
                           "bid": p.get("bid", 0.0)})

    def pop_proposal(self):
        return self.queue.pop(0) if self.queue else None


class AttentionUpdateConsumer(Consumer):
    """Closes the K4 learning loop THROUGH broadcast: feedback envelopes
    carrying per-channel utility drive arbitrator.update_gains(). The
    arbitrator is referenced here only to receive its own learning
    signal — nothing else in the tick touches the gains."""

    name = "attention_update"

    def __init__(self, arbitrator):
        super().__init__()
        self.arbitrator = arbitrator
        self.updates_applied = 0

    def _process(self, envelope):
        p = envelope["payload"]
        utility = p.get("utility")
        if utility:
            self.arbitrator.update_gains(utility,
                                         reward_baseline=p.get("reward_baseline", 0.0))
            self.updates_applied += 1


class ConsolidationEligibleConsumer(Consumer):
    """Marks item ids eligible for later consolidation."""

    name = "consolidation_eligible"

    def __init__(self):
        super().__init__()
        self.eligible: set[str] = set()

    def _process(self, envelope):
        if envelope["payload"].get("consolidation_candidate"):
            self.eligible.add(envelope["item_id"])


class ReportConsumer(Consumer):
    """Accumulates report lines from broadcast content."""

    name = "report"

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def _process(self, envelope):
        p = envelope["payload"]
        self.lines.append(
            f"[t={envelope['tick']}] {envelope['kind']}:{envelope['item_id']} "
            f"channel={p.get('channel')} bid={p.get('bid', 0.0):.3f}")


def build_consumers(arbitrator) -> dict:
    """Instantiate all six consumers, keyed by registry name."""
    return {
        "memory_admit": MemoryAdmissionConsumer(),
        "self_model_update": SelfModelUpdateConsumer(),
        "planner_input": PlannerInputConsumer(),
        "attention_update": AttentionUpdateConsumer(arbitrator),
        "consolidation_eligible": ConsolidationEligibleConsumer(),
        "report": ReportConsumer(),
    }
