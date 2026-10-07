# H9 — chroma (D6) donor adoption — integration specification

**Package:** `flesh-pits/handoff/packages/H9` (adoption record + fetch.sh;
donor NOT vendored)
**Maturity:** ADOPT. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is fetched, installed, or
merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

- **Donor:** chroma-core/chroma — https://github.com/chroma-core/chroma
- **Pinned SHA:** `f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f` (commit 2026-10-06)
- **License:** Apache-2.0 (raw LICENSE at SHA, VERIFIED per donor_matrix D6)
- **Primary surface:** the primary's retrieval layer — episodic/semantic
  retrieval substrate, LOCAL mode. The scratch copy shows the primary's
  current retrieval as ad-hoc vector code (`core/memory_integration.py`
  `CanonicalMemory.retrieve(...)`, `core/memory_port/experience_store.py`
  `get_similar_experiences(...)` — exact live-tree shape UNVERIFIED, Q7).
- **Seam contract:** Chroma sits BEHIND a retrieval interface the primary
  owns. The primary defines `retrieve(query_vector, k, filters) →
  [records]`; Chroma's `Collection.query()` is one implementation of that
  interface. The current substrate stays as the fallback implementation.
  Do not let Chroma's API leak into callers — the interface boundary is
  what makes rollback trivial.

## 2. Interface contract

Fetch (nothing vendored): `./fetch.sh [dest-dir]` clones and checks out
the pinned SHA detached. Mechanism: embedding store with HNSW
approximate-nearest-neighbor retrieval; collections, metadata filtering,
pluggable embedding functions.

Adapter responsibilities (primary lane):
- **Interface first:** define the primary's retrieval interface
  (`query_vector: list[float], k: int, metadata_filters: dict →
  list[{id, vector, payload, distance}]`) BEFORE touching Chroma.
  Implement it twice: current substrate (baseline) and Chroma local-mode
  collection (candidate). Both behind a config flag.
- **Embedding function:** pluggable per Chroma's design — the primary's
  existing embedding path (whatever produces `context_vector` today)
  becomes the embedding function. **Index rebuild on embedding-model
  change** is a known operational cost: version the index by
  (embedding-model-id, chroma-SHA) and rebuild, never mutate in place.
- **Collections:** one collection per memory class the primary
  distinguishes (episodic vs semantic at minimum, if the primary keeps
  that distinction) — do not flatten them into one index without a
  benchmark showing no recall loss.
- **Metadata:** provenance fields (origin_class, timestamps from H6)
  ride as Chroma metadata for filtered retrieval — this composes with
  H6 rather than duplicating it.
- **State ownership:** the Chroma persistent directory is primary-owned
  state; backup/restore story is the primary's (HNSW index files are not
  human-diffable — keep the source records as the system of record).

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: standardized vector retrieval substrate; low integration
cost; maintained upstream. NOT a claimed recall improvement — retrieval
quality is deployment-dependent (manifest bound).

Preregistration (stranger-runnable):
1. **Record validation:** `python3 tests/test_h9.py` in the package —
   validates the adoption record (pin format, license, fetch script
   executable). PASS required before fetching.
2. **Fetch integrity:** run `./fetch.sh /tmp/chroma-d6`; gate:
   `git rev-parse HEAD` == `f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f`.
3. **Benchmark before switching (the actual gate):** on the primary's own
   recall/association workload, measure recall@k and p99 query latency
   for (a) current substrate, (b) Chroma local mode, same embedding
   function, same corpus. Gate: Chroma recall@k ≥ current − 2%
   AND p99 latency ≤ current × 1.5, on a preregistered probe set.
   If it fails, the substrate stays — do not switch on reputation.
4. **Filter correctness:** metadata-filtered queries return exactly the
   filtered set (no HNSW approximation leakage across the filter
   boundary) — gate: 100% on a synthetic probe.

## 4. Rollback + tripwire

- **Rollback:** keep the current substrate — revert the interface binding
  flag to the previous implementation (per manifest). Chroma's data dir
  can be deleted; source records remain the system of record.
- **Tripwire (proves harm):** a scheduled recall@k probe (the benchmark
  from test 3, frozen). If recall@k drops > 5 points vs the acceptance
  baseline, or p99 latency exceeds 2x baseline for a day, auto-revert the
  binding flag and alert. Suspect causes in order: embedding-model drift
  without index rebuild; HNSW memory pressure at scale (manifest risk);
  corpus distribution shift.

## 5. Bounds and risks (carried over verbatim, not softened)

- Retrieval quality is deployment-dependent — benchmark against the
  current substrate before switching.
- Index internals not re-verified at this SHA (well-established upstream;
  UNVERIFIED at this SHA).
- Index rebuild on embedding-model change; memory footprint of HNSW at
  scale.
- Model dependencies: Python; hnswlib/onnxruntime; optional server mode
  (local mode is the adopted surface — server mode is out of scope).
