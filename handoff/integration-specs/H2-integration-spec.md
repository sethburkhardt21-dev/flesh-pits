# H2 — bounded workspace buffer + sole-path broadcast — integration specification

**Package:** `flesh-pits/handoff/packages/H2` (src sha256 in `manifest.json`)
**Maturity:** REPRODUCED. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

Seam identity from the lab-local scratch copy (live tree UNVERIFIED — Q2):

- **Primary modules:**
  - `core/cognitive_workspace.py` — `Workspace.select(concerns, ...)` (ASSUMED;
    the current workspace model — pressure-based selection, no explicit
    capacity bound visible in the scratch copy)
  - `core/broadcast_integration.py` — `CanonicalBroadcast.broadcast_concerns(...)`,
    with consumers `MemoryAdmitConsumer`, `SelfModelConsumer`, `PlannerConsumer`,
    `AttentionConsumer`, `ConsolidationConsumer` (each with `receive(envelope)`)
- **ASSUMED seam:** wrap `CanonicalBroadcast.broadcast_concerns` with a
  `BroadcastBus` in which the broadcast is the ONLY consumer path:
  consumers are reachable exclusively via `bus.broadcast()`; the bus
  validates declared consumers against a registry, receipts every delivery,
  and supports `lesion_consumer` / `lesion_all` / `restore_*`.
  Wrap `Workspace` admission with `BoundedWorkspace(capacity=K)` so the
  buffer becomes a causal bottleneck (finite K → competition → interference,
  the K1 kill experiment).

This is the largest architectural of the mechanism specs — it changes the
primary's internal topology (adds a bottleneck by design). The manifest's
own risk note applies: "adds a bottleneck; the burden of proof (K1/K2) must
travel with the pattern."

## 2. Interface contract

Vendored: `src/workspace_buffer.py` (`BoundedWorkspace`, `WorkspaceItem`,
`BufferRefused`, `admit_swallowing_refusal`, policies
`lowest_bid_oldest_tiebreak`/`oldest`), `src/broadcast.py`
(`BroadcastBus`, `BroadcastRefused`, default `CONSUMER_REGISTRY`).

```python
class BoundedWorkspace:
    def __init__(self, capacity: int | None, eviction_policy=LOWEST_BID_OLDEST_TIEBREAK,
                 ttl_ticks: int = 2, consumer_registry: set[str] | None = None)
    def admit(self, kind, payload: dict, bid: float, declared_consumers,
              ttl_ticks=None, tick=None) -> dict
        # receipt: {item_id, kind, admitted, evicted_item_id|None, live_count, reason, tick}
        # raises BufferRefused on contract violation (bad bid, unknown consumer)
        # raises _NewcomerEvicted (internal) when the newcomer loses — the
        #   newcomer is NOT protected: low-bid newcomers against full
        #   high-bid residents are evicted immediately
    def tick(self) -> dict        # decays items older than ttl_ticks
    def contents(self) -> list    # live items, admission order
    def drain(self) -> list       # post-broadcast clearing
    def stats(self) -> dict
```

```python
class BroadcastBus:
    def __init__(self, registry: dict | None = None)   # defaults to the 6-consumer CONSUMER_REGISTRY
    def register(self, consumer: str, handler, *, state_provider=None)
    def read_consumer(self, consumer: str)              # sanctioned reverse read (consumer -> tick)
    def broadcast(self, item_id, kind, payload: dict, declared_consumers, tick: int) -> dict
        # fan-out to EXACTLY the declared, unlesioned consumers; each delivery
        # receipted; payload deep-copied per consumer (a consumer cannot corrupt the next)
    def lesion_consumer(self, consumer: str) / def lesion_all(self)
    def restore_consumer(self, consumer: str) / def restore_all(self)
```

Adapter responsibilities (primary lane):
- **Consumer mapping:** map the primary's real consumers (memory admit,
  self-model, planner, attention update, consolidation, report) to handler
  functions with the signature `handler(consumer, envelope)`; register each
  exactly once. The lab's registry is a STARTING declaration — the primary
  must assert that each declared consumer has a REAL handler (K2 verifies
  consumption, not declaration).
- **Envelope shape:** `{"item_id", "kind", "payload", "declared_consumers",
  "tick"}` — the primary's existing `broadcast_concerns` envelope can be
  mapped onto this; document the mapping.
- **Capacity K:** primary's choice; the lab ran K=3 for the interference
  kill experiment. K is the competition knob: too small starves consumers,
  too large makes the bound meaningless. Start at the primary's current
  effective fan-out and tighten only under the K1 test.
- **State ownership:** buffer contents and bus deliveries are in-memory and
  per-tick; persistence across restart is the primary's existing concern
  (the lab buffer has TTL decay — restart semantics are primary's call).
- **Error behavior:** `BufferRefused`/`BroadcastRefused` → fail closed
  (refuse the admission/broadcast, receipt it, continue the tick without the
  item). Unknown declared consumers are REFUSED, never silently dropped.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: provable global availability — a broadcast that demonstrably
changes downstream consumers. Lab evidence: K1 interference index
0 / 0.507 / 1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf;
K2 total lesion silences all six consumers at once, internal processing
continues, full recovery, no side channels.

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H2 /tmp/pkgtest && python3
   tests/test_h2.py` — must pass K1+K2 exactly (0/0.507/1.0; lesion/recovery).
2. **K2 sole-path test on the primary (the acceptance gate):** with the bus
   wired, call `lesion_all()` mid-run: ALL primary consumers must go silent
   at once (zero deliveries receipted), internal processing must continue
   (buffer still admits, ticks still advance), and `restore_all()` must
   restore full function with no residual effects. Then `lesion_consumer(x)`
   for each consumer individually: only that consumer goes silent.
   Gate: 100% of the primary's declared consumers behave this way. If any
   consumer still receives content during `lesion_all()`, a side channel
   exists — REJECT the claim, keep the adapter, fix the path.
3. **K1 capacity lesion:** with the bounded buffer at K=3, inject distractor
   items at graded bids; measure the interference index on a primary task
   metric (lab's: target-attention accuracy degradation). Gate: monotonic
   interference in distractor bid, ~0 at K=inf. Proves the buffer is a
   CAUSAL bottleneck, not decoration.
4. **Baseline comparison:** unbounded queue / direct-call wiring must show
   bit-identical trajectories to the direct-call baseline on a fixed probe
   (Phase-4 battery 001b: Delta=0 theater — prove the new path changes
   NOTHING before the capacity bound is engaged).

## 4. Rollback + tripwire

- **Rollback:** remove the bus, restore direct calls (the pre-bus wiring is
  the trivial fallback per the manifest). The buffer degrades to
  `capacity=None` (unbounded) as an intermediate step — K=inf removes the
  competition without removing the receipts.
- **Tripwire (proves harm):** run the K2 gate on every scheduled integration
  test. If `lesion_all()` ever fails to silence all consumers (a new side
  channel was introduced), or if `evicted_total` grows unboundedly with
  task-relevant items being evicted (measure: % of evicted items with
  bid > median admitted bid — above 25% sustained = the bottleneck is
  eating the task), page the primary lane. Do NOT silently raise K to make
  the alarm stop — that is evidence the bound is wrong for the workload.

## 5. Bounds and risks (carried over verbatim, not softened)

- Adds a bottleneck; the burden of proof (K1/K2) must travel with the
  pattern — re-run K2 against the primary's consumer set or drop the claim.
- REPRODUCED on 5/5 fresh seeds; single lab — no independent replication
  yet (stated, not hidden).
- Broadcast without measured consumer change is theater (the lab's
  anti-theater law). If the K2 gate cannot be made to pass on the primary,
  do not claim a global workspace — the pattern stays in the lab.
