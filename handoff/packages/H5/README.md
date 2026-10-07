# H5 — consolidation_cycle (offline priority consolidation) — S-01 winner

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** CAUSAL (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

offline priority consolidation pass over the episodic store: decay -> merge (relevance-desc greedy seed, cosine to seed) -> prune (summary layer only, input never mutated). S-01's 'promote ids, replay raw' beat memory_port's 'replay compressed summaries' 4/4

## Baseline

no consolidation

## Key results

- CAUSAL (Phase-4 battery) — evidence HISTORICAL, stated as such
  - Receipt: `Phase-4 battery receipts`
- EXP-FP-0005 race (CLOSED 2026-10-07, 4 fresh seeds, preregistered, all gates passed): S-01 wins 4/4 (Delta=+0.113 seed-mean IG)
  - Receipt: `receipts/EXP-FP-0005-S.json (vendored); receipts/EXP-FP-0005-D.json (vendored)`
- package test proves the mechanism standalone (cycle, determinism, immutability, merge correctness, verification flow, fail-closed)
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

battery consolidation ablation (Phase-4); EXP-FP-0005 arms P1/P2/U/N

## Generalization bounds

- RACE BOUND (honest): neither implementation's offline replay beats no-replay on the pomaze corpus (P1-vs-N 2/4, P2-vs-N 1/4 per-seed wins; seed-mean IG +0.049/-0.064 vs N 0.0) — mechanism behaves per docs, no measured offline performance lift on this corpus
- PE-magnitude prioritization is actively WORSE than uniform replay here (uniform wins 4/4, seed-mean +0.323) — do not wire priority replay without per-deployment validation
- not generalized beyond the battery task; priority function needs validation per deployment

## Integration surface

any episodic store on the primary with an offline window

## Expected benefit

measurable offline improvement of retrieval quality (claimed with the race bound above — performance lift UNPROVEN on current evidence)

## Risks

- priority function is deployment-specific; the race shows PE-magnitude prioritization hurts vs uniform on pomaze

## Rollback

disable the consolidation pass

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/consolidation_cycle_s01.py` | mechanism | `c75e3892b90973c3` |
| `src/consolidation_cycle_memory_port.py` | race loser (reference; the memory_port variant that lost 4/4) | `4f6707cc102310e0` |
| `tests/test_h5.py` | package test (preregistered procedure) | `3c421e2e48e355dc` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H5 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h5.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- vendored files are byte-identical to the proven originals (PROVENANCE.md in race_vendored/), NOT modified
- maturity ledger lists the race as EXECUTED (negative result) — the mechanism itself is CAUSAL per the battery

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
