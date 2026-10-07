"""Architecture A — broadcast bus: the SOLE path to consumers.

Load-bearing claim 3: workspace content reaches consumers ONLY through
this bus. The tick module is constructed so consumers are reachable
exclusively via ``BroadcastBus.broadcast()`` — no consumer references
exist anywhere else in the tick path (enforced by construction, verified
by the K2 lesion: drop the whole bus and every consumer must go silent
at once).

Semantics (ported from donor malice_port/broadcast.py, in-memory):
  - consumers are declared in CONSUMER_REGISTRY;
  - each workspace item declares its consumer set at admission;
  - fan-out delivers to exactly the declared, unlesioned consumers;
  - every delivery and lesion skip is receipted in memory.
Lesions (experiment, not failure):
  - lesion_consumer(name): drop one consumer from fan-out (K2 selective);
  - lesion_all(): silence the entire broadcast (K2 total);
  - restore_*: reverse. Lesion history is preserved for audit.
"""
from __future__ import annotations

import copy

# Declared consumers of Architecture A. Each must map to a real consumer
# implementation in consumers.py (K2 verifies consumption, not declaration).
CONSUMER_REGISTRY = {
    "memory_admit": "workspace -> episodic memory admission",
    "self_model_update": "self-state vector update",
    "planner_input": "planning queue input",
    "attention_update": "learned attention gain update (K4 hook)",
    "consolidation_eligible": "marks content eligible for consolidation",
    "report": "report channel accumulation",
}


class BroadcastRefused(ValueError):
    """Fail-closed: unknown consumer, undeclared fan-out, bad envelope."""


class BroadcastBus:
    def __init__(self, registry=None):
        self.registry = dict(registry) if registry else dict(CONSUMER_REGISTRY)
        self._handlers: dict[str, callable] = {}
        self._state_providers: dict[str, callable] = {}
        self._lesioned: set[str] = set()
        self._bus_lesioned = False          # lesion_all(): the whole bus
        self.deliveries: list[dict] = []    # in-memory delivery receipts
        self.lesion_log: list[dict] = []    # audit trail of lesions
        self.broadcast_count = 0

    # ---------------------------------------------------------------
    def register(self, consumer: str, handler, *, state_provider=None) -> None:
        if consumer not in self.registry:
            raise BroadcastRefused(f"unknown consumer {consumer!r}")
        if consumer in self._handlers:
            raise BroadcastRefused(f"consumer {consumer!r} already registered")
        self._handlers[consumer] = handler
        if state_provider is not None:
            self._state_providers[consumer] = state_provider

    def read_consumer(self, consumer: str):
        """Sanctioned reverse read: the tick may read a consumer's state
        through the bus (e.g. action selection draining the planner
        queue). Workspace CONTENT still reaches consumers only via
        broadcast(); this direction is consumer -> tick and is mediated
        by the bus, not a side reference."""
        provider = self._state_providers.get(consumer)
        if provider is None:
            raise BroadcastRefused(
                f"consumer {consumer!r} exposes no state through the bus")
        return provider()

    def broadcast(self, item_id: str, kind: str, payload: dict,
                  declared_consumers, tick: int) -> dict:
        """Fan-out one workspace item. Returns delivery evidence."""
        declared = list(declared_consumers)
        unknown = [c for c in declared if c not in self.registry]
        if unknown:
            raise BroadcastRefused(f"declared unknown consumers: {unknown}")
        self.broadcast_count += 1
        delivered, skipped = [], []
        for consumer in declared:
            if self._bus_lesioned or consumer in self._lesioned:
                self.deliveries.append({
                    "item_id": item_id, "kind": kind, "consumer": consumer,
                    "status": "lesioned", "tick": tick})
                skipped.append(consumer)
                continue
            handler = self._handlers.get(consumer)
            if handler is None:
                raise BroadcastRefused(
                    f"consumer {consumer!r} declared but no handler registered")
            envelope = {"item_id": item_id, "kind": kind,
                        "payload": copy.deepcopy(payload),
                        "declared_consumers": declared, "tick": tick}
            handler(consumer, envelope)     # a consumer that mutates its
                                            # copy cannot corrupt the next
            self.deliveries.append({
                "item_id": item_id, "kind": kind, "consumer": consumer,
                "status": "delivered", "tick": tick})
            delivered.append(consumer)
        return {"item_id": item_id, "delivered": delivered,
                "lesioned_skips": skipped, "declared": declared}

    # --------------------------------------------------------------- lesions
    def lesion_consumer(self, consumer: str) -> None:
        if consumer not in self.registry:
            raise BroadcastRefused(f"unknown consumer {consumer!r}")
        self._lesioned.add(consumer)
        self.lesion_log.append({"action": "lesion_consumer",
                                "consumer": consumer})

    def lesion_all(self) -> None:
        self._bus_lesioned = True
        self.lesion_log.append({"action": "lesion_all"})

    def restore_consumer(self, consumer: str) -> None:
        self._lesioned.discard(consumer)
        self.lesion_log.append({"action": "restore_consumer",
                                "consumer": consumer})

    def restore_all(self) -> None:
        self._lesioned.clear()
        self._bus_lesioned = False
        self.lesion_log.append({"action": "restore_all"})

    @property
    def lesioned(self):
        return set(self._lesioned)

    @property
    def bus_lesioned(self):
        return self._bus_lesioned
