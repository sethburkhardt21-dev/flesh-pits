# H9 — chroma (D6) adoption record

- **Repo / URL:** chroma-core/chroma — https://github.com/chroma-core/chroma
- **Pinned SHA:** `f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f` (commit 2026-10-06)
- **License:** Apache-2.0 (raw LICENSE at SHA, VERIFIED per donor_matrix D6)
- **License note:** permissive; attribution only.

## Mechanism
Embedding store with **HNSW** approximate-nearest-neighbor retrieval;
collections, metadata filtering, pluggable embedding functions. Simple,
fast vector database for AI apps; local/server modes.

## Disposition (donor_matrix D6): ADOPT
Standardize vector retrieval on Chroma (local mode) as the
episodic/semantic **retrieval** substrate for the primary. Low integration
cost (pip install, local mode, pluggable embedders); maintained;
permissive license.

## Integration surface
The primary's retrieval layer — replaces ad-hoc vector stores behind the
episodic/semantic retrieval interface.

## Benchmark plan (before switching)
Retrieval latency/recall baseline: benchmark Chroma (local mode) against
the primary's current substrate on the primary's own recall/association
workload. Retrieval quality is deployment-dependent — do not switch on
reputation; switch on measured recall/latency.

## Risks
Minimal. Standard operational caveats: index rebuild on embedding-model
change; memory footprint of HNSW at scale. Index internals not re-verified
at this SHA (well-established upstream; marked UNVERIFIED at this SHA).

## Rollback
Keep the current substrate. Chroma sits behind the retrieval interface;
revert the interface binding to the previous implementation.

## Fetch
`./fetch.sh` clones the repo and checks out the pinned SHA (detached HEAD).
Nothing is vendored in this package — the donor is fetched at adoption time
from the pinned commit.
