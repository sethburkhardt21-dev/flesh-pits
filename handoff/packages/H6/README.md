# H6 — memory_provenance hash-chaining

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** CAUSAL (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

hash-chained ProvenanceStore on memory writes (retention law, compare-and-set rollback, path allowlist)

## Baseline

unchained writes

## Key results

- CAUSAL, tick-fresh in live var/being (Phase-1 audit)
  - Receipt: `Phase-1 audit evidence`
- package test proves the chaining standalone (bootstrap degrades, retention, tamper-breaks-chain, rollback, allowlist, store-tamper detection)
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

Phase-1 audit evidence; fail-open tick / fail-closed claims

## Generalization bounds

- 'agent' provenance origin unreachable via the public API (design gap — INCONCLUSIVE item; the test seeds the genesis record to exercise the retention rule)

## Integration surface

memory write paths

## Expected benefit

tamper-evident memory writes

## Risks

- 'agent' provenance genesis path needs architect attention

## Rollback

n/a — already live

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/memory_provenance.py` | mechanism (lab-local copy; see notes) | `bff6f99c1d2548e7` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H6 /tmp/pkgtest && cd /tmp/pkgtest
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- harvest entry lists the source as the live tree (core/memory_port/memory_provenance.py, already live-wired); this package was built from the lab-local copy at flesh-pits/var/scratch-mneumora-main/mneumora/core/memory_port/memory_provenance.py (inside flesh-pits; the live Being tree was not touched). Canonical source remains the live tree.
- listed for completeness per §51 reorient

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
