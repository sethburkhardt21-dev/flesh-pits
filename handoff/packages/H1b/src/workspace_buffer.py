"""Architecture A — true bounded in-memory workspace buffer.

Replaces the JSON-file shim model (donor malice_port/workspace.py) with a
LIVE in-memory buffer: capacity K items, an explicit eviction policy, and
per-admission receipts. This buffer is the causal bottleneck of
Architecture A: with finite K, simultaneous high-salience items compete
and interfere (K1 kill experiment).

Cognitive neutrality: items are opaque (kind, payload, bid); no
identity-privileged machinery exists anywhere in this module.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# eviction policies
# ---------------------------------------------------------------------------
LOWEST_BID_OLDEST_TIEBREAK = "lowest_bid_oldest_tiebreak"  # default
OLDEST = "oldest"

POLICIES = (LOWEST_BID_OLDEST_TIEBREAK, OLDEST)


@dataclass(frozen=True)
class WorkspaceItem:
    """One admitted unit of workspace content."""
    item_id: str
    kind: str
    payload: dict
    bid: float                       # gained salience bid at admission
    admitted_tick: int
    declared_consumers: tuple        # validated against the broadcast registry
    ttl_ticks: int = 2               # stale content decays out of the present


class BufferRefused(ValueError):
    """Admission contract violated; no state changed."""


class BoundedWorkspace:
    """Capacity-K live buffer with explicit eviction.

    Parameters
    ----------
    capacity : int | None
        Max live items. ``None`` = unbounded (the K1 K=infinity condition).
    eviction_policy : str
        ``lowest_bid_oldest_tiebreak`` (default): when over capacity, evict
        the lowest-bid item; ties broken by oldest admission (documented,
        deterministic). ``oldest``: pure FIFO eviction.
    ttl_ticks : int
        Items older than ``ttl_ticks`` ticks are decayed out on ``tick()``.
    consumer_registry : set[str] | None
        Declared consumers are validated against this at admission
        (fail-closed: unknown consumers refused, mirroring the donor
        broadcast semantics, but in memory).
    """

    _ids = itertools.count(1)

    def __init__(self, capacity, eviction_policy=LOWEST_BID_OLDEST_TIEBREAK,
                 ttl_ticks=2, consumer_registry=None):
        if capacity is not None and (not isinstance(capacity, int) or capacity < 1):
            raise BufferRefused("capacity must be a positive int or None (unbounded)")
        if eviction_policy not in POLICIES:
            raise BufferRefused(f"unknown eviction policy {eviction_policy!r}")
        if not isinstance(ttl_ticks, int) or ttl_ticks < 1:
            raise BufferRefused("ttl_ticks must be a positive int")
        self.capacity = capacity
        self.eviction_policy = eviction_policy
        self.ttl_ticks = ttl_ticks
        self.consumer_registry = set(consumer_registry) if consumer_registry else set()
        self._items: dict[str, WorkspaceItem] = {}
        self._order: list[str] = []          # admission order (deterministic tie-break)
        self.tick_index = 0
        self.admitted_total = 0
        self.evicted_total = 0
        self.evicted_ids: list[str] = []     # causal audit: what competition killed
        self.receipts: list[dict] = []

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------
    def _validate(self, item: WorkspaceItem) -> None:
        if not isinstance(item.bid, (int, float)) or not (item.bid == item.bid):
            raise BufferRefused(f"bid must be a finite number, got {item.bid!r}")
        if item.bid != item.bid or abs(item.bid) == float("inf"):
            raise BufferRefused(f"bid must be finite, got {item.bid!r}")
        if self.consumer_registry:
            unknown = [c for c in item.declared_consumers
                       if c not in self.consumer_registry]
            if unknown:
                raise BufferRefused(f"declared unknown consumers: {unknown}")
            if not item.declared_consumers:
                raise BufferRefused("item must declare at least one consumer")

    def admit(self, kind, payload, bid, declared_consumers,
              ttl_ticks=None, tick=None) -> dict:
        """Admit one item; evict under policy if over capacity.

        Returns a receipt: {item_id, admitted, evicted_item_id or None,
        live_count, reason}. Deterministic.
        """
        tick = self.tick_index if tick is None else tick
        item = WorkspaceItem(
            item_id=f"ws-{next(self._ids)}",
            kind=str(kind),
            payload=dict(payload),
            bid=float(bid),
            admitted_tick=tick,
            declared_consumers=tuple(declared_consumers),
            ttl_ticks=self.ttl_ticks if ttl_ticks is None else int(ttl_ticks),
        )
        self._validate(item)

        evicted_id = None
        if self.capacity is not None and len(self._items) >= self.capacity:
            evicted_id = self._evict(item)
        self._items[item.item_id] = item
        self._order.append(item.item_id)
        self.admitted_total += 1
        receipt = {"item_id": item.item_id, "kind": item.kind,
                   "admitted": True, "evicted_item_id": evicted_id,
                   "live_count": len(self._items),
                   "reason": "eviction" if evicted_id else "capacity_available",
                   "tick": tick}
        self.receipts.append(receipt)
        return receipt

    def _evict(self, newcomer: WorkspaceItem) -> str:
        """Choose and remove one item under the eviction policy.

        NOTE: the newcomer competes on the same terms as residents — it is
        NOT protected. A low-bid newcomer against full high-bid residents
        is itself evicted immediately (target interference, K1).
        """
        if self.eviction_policy == OLDEST:
            victim = self._order[0]
        else:
            candidates = [(self._items[i].bid, self._items[i].admitted_tick, i)
                          for i in self._order] + \
                         [(newcomer.bid, newcomer.admitted_tick, newcomer.item_id)]
            # lowest bid loses; tie -> oldest admission; tie -> canonical id order
            victim = min(candidates, key=lambda c: (c[0], c[1], c[2]))[2]
        if victim == newcomer.item_id:
            # newcomer lost the competition: it never enters
            self.evicted_total += 1
            self.evicted_ids.append(victim)
            raise _NewcomerEvicted(newcomer.item_id)
        del self._items[victim]
        self._order.remove(victim)
        self.evicted_total += 1
        self.evicted_ids.append(victim)
        return victim

    # ------------------------------------------------------------------
    # tick / decay / drain
    # ------------------------------------------------------------------
    def tick(self) -> dict:
        """Advance one tick; decay stale items. Returns decay receipt."""
        self.tick_index += 1
        stale = [i for i in self._order
                 if self.tick_index - self._items[i].admitted_tick >= self._items[i].ttl_ticks]
        for i in stale:
            del self._items[i]
            self._order.remove(i)
        return {"tick": self.tick_index, "decayed": stale,
                "live_count": len(self._items)}

    def contents(self) -> list[WorkspaceItem]:
        """Live items, admission order (oldest first). Read-only."""
        return [self._items[i] for i in self._order]

    def drain(self) -> list[WorkspaceItem]:
        """Remove and return all live items (post-broadcast clearing)."""
        items = self.contents()
        self._items.clear()
        self._order.clear()
        return items

    def stats(self) -> dict:
        return {"tick": self.tick_index, "live": len(self._items),
                "capacity": self.capacity, "policy": self.eviction_policy,
                "admitted_total": self.admitted_total,
                "evicted_total": self.evicted_total}


class _NewcomerEvicted(Exception):
    """Internal control flow: the newcomer lost the admission competition."""


def admit_swallowing_refusal(buffer: BoundedWorkspace, *args, **kwargs) -> dict:
    """Admit, but convert a newcomer-eviction into a refusal receipt
    instead of raising (competition is normal, not exceptional)."""
    try:
        return buffer.admit(*args, **kwargs)
    except _NewcomerEvicted as e:
        receipt = {"item_id": str(e), "admitted": False,
                   "evicted_item_id": str(e), "live_count": len(buffer._items),
                   "reason": "newcomer_lost_competition", "tick": buffer.tick_index}
        buffer.receipts.append(receipt)
        buffer.evicted_total += 0  # already counted in _evict
        return receipt
