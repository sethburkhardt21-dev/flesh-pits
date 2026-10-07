# H7 — Experiment harness (preregistration + hash-chained receipts)

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** EXECUTED (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

preregistered experiments (hypothesis, null, metric, baseline, ablation, procedure, seed, interpretation, limitations), canonical episode loop, verify_chain() hash-chained receipts

## Baseline

ad-hoc scripts

## Key results

- proven on the full K-series; 9+8 receipts in Phase 4; K-series + repro + K8 receipts in Phase 5
  - Receipt: `flesh-pits/receipts/ (K4-CANONICAL-RERUN.ndjson, EXP-FP-0005-*.json, k8/k10 receipts)`
- caught NR-A-008 (mis-specified metric) and NR-A-010 (harness artifact) honestly
  - Receipt: `lab negative-results record`
- package test proves the metrology standalone (episode loop, preregistration schema, chained receipts, tamper evidence)
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

n/a (metrology)

## Generalization bounds

- used across A, B, D, and the transfer variant — the lab's shared metrology

## Integration surface

any future primary-side experiment

## Expected benefit

every future claim arrives with a receipt; the K8 wire-by-decision-receipt pattern is the wiring template

## Risks

- two harness variants exist (flesh-pits + brothel) — consolidate to one before primary adoption

## Rollback

n/a (metrology)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/harness.py` | mechanism | `59dd0b98c57054bf` |
| `src/env_interface.py` | dependency | `2b8d026e1bb4688c` |
| `tests/test_h7.py` | package test (preregistered procedure) | `22958fe37ed465b3` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H7 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h7.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- lab env/agent zoo (envs, baselines, arch_d) NOT vendored; the package test stubs them to prove the metrology machinery — stated in the test and README
- brothel twin exists (not vendored)

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
