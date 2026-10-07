# H9 — chroma (D6) — donor adoption

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** ADOPT (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** Apache-2.0 (raw LICENSE at SHA, VERIFIED per donor_matrix).
**Model dependencies:** Python; hnswlib/onnxruntime; optional server mode.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

embedding store with HNSW approximate-nearest-neighbor retrieval; collections, metadata filtering, pluggable embedding functions

## Baseline

the primary's current (ad-hoc) retrieval substrate

## Key results

- donor_matrix D6: ADOPT — repo/API surface verified at docs level; HNSW-based retrieval per documentation
  - Receipt: `emergent-mind/research/donor_matrix.md (D6 entry)`
- package test validates the adoption record standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

n/a (donor)

## Generalization bounds

- retrieval quality is deployment-dependent — benchmark against the current substrate before switching
- index internals not re-verified at this SHA (well-established upstream; UNVERIFIED at this SHA)

## Integration surface

the primary's retrieval layer (episodic/semantic retrieval substrate, local mode)

## Expected benefit

standardized vector retrieval substrate; low integration cost; maintained; permissive license

## Risks

- index rebuild on embedding-model change; memory footprint of HNSW at scale

## Rollback

keep the current substrate (Chroma sits behind the retrieval interface)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `ADOPTION.md` | adoption record | `825252b1d66609b4` |
| `fetch.sh` | fetch script (pins SHA) | `f69c8dfa4099e45a` |
| `tests/test_h9.py` | package test (preregistered procedure) | `a021e290390643d5` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H9 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h9.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- donor NOT vendored (large); fetch.sh pins the exact SHA
- pinned SHA f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f (2026-10-06)

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
