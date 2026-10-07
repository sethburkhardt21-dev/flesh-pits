# H5 — consolidation_cycle S-01 (offline priority consolidation) — integration specification

**Package:** `flesh-pits/handoff/packages/H5` (src sha256 in `manifest.json`)
**Maturity:** CAUSAL (mechanism). **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

Seam identity from the scratch copy (live tree UNVERIFIED — Q5):

- **Primary modules:**
  - `core/memory_integration.py` — `CanonicalMemory.consolidate()`,
    `maybe_consolidate()`, `replay_batch(k=16)` (the current consolidation
    entry points; scratch copy shows `consolidate()` at line ~497)
  - `core/memory_port/consolidation_cycle.py` — `ConsolidationCycle`
    (the RACE LOSER is already live on the primary; the race tested
    S-01's "promote ids, replay raw" against this "replay compressed
    summaries" variant — S-01 won 4/4)
  - `core/memory_port/experience_store.py` — `ExperienceStore`
    (the snapshot source)
- **ASSUMED seam:** an offline scheduler (the primary's existing
  consolidation cadence, wherever `maybe_consolidate()` is called from)
  takes a snapshot of the episodic store, runs
  `ConsolidationCycle.run_over_episodes(episodes, tick=tick)`, and:
  1. writes the returned `summaries` into the summary layer with status
     UNVERIFIED,
  2. hands `promotion` ids to the existing `replay_batch()` path,
  3. requires `CycleResult.mark_verified(summary_id, evidence)` by an
     INDEPENDENT verifier before summaries are consumed downstream.
- The module is deliberately store-agnostic: it takes a caller-supplied
  list, never touches the write path, never mutates input, never deletes
  source episodes. It "cannot be wired into a hot path except by a
  scheduler explicitly calling run()" — preserve that property; do NOT
  inline it into `admit()`.

## 2. Interface contract

Vendored: `src/consolidation_cycle_s01.py` (`ConsolidationCycle`,
`CycleConfig`, `CycleResult`, `CycleRefused`, `adapt_episode`,
`cosine_similarity`, `UNVERIFIED`/`VERIFIED`).
Also vendored: `src/consolidation_cycle_memory_port.py` — the RACE LOSER,
reference only (do not wire; it lost 4/4).

```python
@dataclass(frozen=True)
class CycleConfig:
    decay_rate: float = 0.99        # (0, 1]  — multiplicative relevance decay
    merge_threshold: float = 0.9    # (0, 1]  — cosine-to-seed join threshold
    prune_threshold: float = 0.1    # [0, 1)  — summary-layer prune floor
    relevance_floor: float = 0.2    # [0, 1]  — seeded relevance for new entries
    min_episodes: int = 1           # below this the cycle does no work (ran=False)
    max_summaries: int = 256       # overflow REPORTED, never silently dropped
    promotion_k: int = 16

class ConsolidationCycle:
    def __init__(self, config: CycleConfig | None = None)
    def run(self, entries: Sequence[Mapping], *, tick: int) -> CycleResult
        # entries: {"id": str, "vector": [finite floats], "relevance": [0,1], ...free-form}
        # pipeline: decay -> greedy merge (relevance-desc, id-asc; cosine to SEED > threshold)
        #           -> prune (summary layer only) -> promotion (top-k by relevance)
        # deterministic: NO hash-order tie-breaks anywhere; CycleRefused on bad input
    def run_over_episodes(self, episodes, *, tick: int) -> CycleResult
        # episodes: {"episode_id", "context_vector", "salience", ...} per adapt_episode

class CycleResult:
    summaries: list[dict]   # each {summary_id, member_ids, merged_count, vector,
                            #   relevance, status: UNVERIFIED, tick} — UNVERIFIED until mark_verified
    overflow: list[dict]    # beyond max_summaries, reported not lost
    promotion: list[str]    # top-k summary ids — replay/promotion candidates
    tick: int; ran: bool
    def mark_verified(self, summary_id: str, evidence: str)  # independent verifier only
    def unverified(self) -> list[dict]
```

Adapter responsibilities (primary lane):
- **Snapshot → entries:** map the primary's episode records to the entry
  shape (`adapt_episode` covers the experience-store shape:
  `episode_id`/`context_vector`/`salience`). Salience→relevance mapping
  must be documented (lab used salience directly).
- **Summary layer:** the primary owns where summaries live; every summary
  starts UNVERIFIED and an independent verifier (not the cycle, not the
  admission path) flips it. Raw episodes stay retained until verification.
- **Promotion → replay:** feed `promotion` ids into the primary's existing
  replay path (`replay_batch`). The race result says "promote ids, replay
  RAW" beats "replay compressed summaries" — wire the winner's pattern,
  not the loser's.
- **Priority function:** the race shows PE-magnitude prioritization is
  ACTIVELY WORSE than uniform replay here (uniform wins 4/4, seed-mean
  +0.323). Do NOT wire PE-priority; the relevance seeding should be
  salience/uniform until a per-deployment priority function is validated.
- **Error behavior:** `CycleRefused` → the cycle does nothing (ran=False
  or raise before work); never partially consolidate.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta — **STATED WITH THE RACE BOUND (see §5):** the mechanism
behaves per docs (decay→merge→prune, determinism, immutability,
verification flow, fail-closed); NO measured offline performance lift vs
no-replay was found on the pomaze corpus. The honest expected benefit is
a well-formed offline consolidation pipeline with receipts, not a
retrieval-quality lift.

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H5 /tmp/pkgtest && python3
   tests/test_h5.py` — cycle/determinism/immutability/merge/verify/
   fail-closed, PASS.
2. **Mechanism probes on the primary's store:** run one cycle over a
   frozen snapshot of the primary's episodic store; assert determinism
   (two runs → identical summary_ids), immutability (input snapshot
   byte-identical after), and that every summary starts UNVERIFIED.
   Gate: all three hold.
3. **The lift question (the honest test):** replicate EXP-FP-0005's race
   on the PRIMARY's corpus — arms: S-01 replay vs no-replay vs uniform
   replay, preregistered metric = retrieval information gain (or the
   primary's retrieval-quality metric), 4 fresh seeds. The package does
   NOT claim this will show a lift; run it to find out. If no lift:
   the consolidation stays as hygiene (bounded store, provenance-rich
   summaries), not as a performance feature — label it accordingly.
4. **Loser quarantine:** the primary's existing
   `memory_port/consolidation_cycle.py` (the race loser) must not be
   re-wired as the consolidation path while S-01 is adopted — or the
   race must be re-run on the primary's corpus to re-decide. No silent
   reversion to the loser.

## 4. Rollback + tripwire

- **Rollback:** disable the consolidation pass (per manifest). The store
  is untouched by design (input never mutated, source episodes never
  deleted), so rollback is removal of the scheduler call. Summaries
  already written stay UNVERIFIED until the verifier dispositions them —
  the primary defines the disposition policy (retain-as-unverified or drop).
- **Tripwire (proves harm):** if post-consolidation retrieval quality on
  the primary's benchmark drops vs the pre-consolidation baseline for two
  consecutive cycles (the race's P2-vs-N pattern: seed-mean −0.064), halt
  the pass and inspect the priority function first — the race says
  PE-priority is the likely culprit. Also: if `overflow` is non-empty for
  three consecutive cycles, `max_summaries` is undersized for the
  workload — raise the cap or shorten the cadence, do not silently drop
  overflow.

## 5. Bounds and risks (carried over verbatim, not softened)

- **RACE BOUND (honest):** neither implementation's offline replay beats
  no-replay on the pomaze corpus (P1-vs-N 2/4, P2-vs-N 1/4 per-seed wins;
  seed-mean IG +0.049/−0.064 vs N 0.0) — mechanism behaves per docs, no
  measured offline performance lift on this corpus.
- **PE-magnitude prioritization is actively WORSE than uniform replay
  here** (uniform wins 4/4, seed-mean +0.323) — do not wire priority
  replay without per-deployment validation.
- Not generalized beyond the battery task; priority function needs
  validation per deployment.
- Vendored files are byte-identical to the proven originals (NOT modified).
- Maturity ledger lists the race as EXECUTED (negative result) — the
  mechanism itself is CAUSAL per the battery. Do not present the race
  win as a lift claim.
