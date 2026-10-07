# H1b — Cue-indexed gain adapter (widens H1 past NR-A-007)

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** CAUSAL (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

the K4/P3 reversal-tracking mechanism operating per cue value across rule flips: CueIndexedArbitrator holds one gain vector per cue value via context-keyed gains property; original delta rule unchanged; tick.py additive arbitrator_cls/context_fn params, defaults = proven H1 behavior; context_fn without set_context fails closed

## Baseline

frozen gains on cue-conditioned changing_rule WITH cue input

## Key results

- K10 B1 (cue in input): R = 1.52/1.68/1.64/1.58, 4/4 WIN (learned 368-392/480 vs frozen 230-244/480)
  - Receipt: `receipts/k10_cue_indexed_adapter.ndjson (vendored); preregistration receipts/prereg_k10_cue_indexed_adapter.json`
- B2 control (cue withheld): R 0.95-1.13, 0/4 gaps — the cue, not the new module, is causal
  - Receipt: `same ndjson`
- per-cue gain vectors diverge as designed (e.g. cue0 {a0:1.925, a1:0.875}, cue1 {a0:0.95, a1:2.0})
  - Receipt: `same ndjson`
- package test reproduces B1 1.52/1.68/1.64/1.58 and B2 0.95-1.13 exactly, standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

B2 IS the ablation (withhold the cue -> advantage vanishes)

## Generalization bounds

- single experiment family; second-lane replication open
- context dimensionality untested beyond binary cue

## Integration surface

any primary selection point with observable context — pass context_fn; H1's loop does the rest

## Expected benefit

lifts H1's architectural bound on context-conditioned tasks; per-cue gain vectors are inspectable

## Risks

- cue must be in the input — without it the bound stands (NR-A-007-without-cue-input)
- context dimensionality untested beyond binary cue

## Rollback

default arbitrator_cls/context_fn = the proven H1 path (one-flag)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/attention_cue.py` | mechanism | `2155b30ebd4debae` |
| `src/tick.py` | mechanism | `ca78fd8ff9e01d5f` |
| `src/attention.py` | dependency | `ee977a1de7f28200` |
| `src/bids.py` | dependency | `292dc50e7553243a` |
| `src/broadcast.py` | dependency | `8db343efa51728e7` |
| `src/workspace_buffer.py` | dependency | `09eda3384fdf1de2` |
| `src/ignition.py` | dependency | `cc38671f2c0dc6cd` |
| `src/consumers.py` | dependency | `44096a6d97951b62` |
| `tests/support/changing_rule.py` | test-support env | `f861a6dac879ee20` |
| `tests/support/env_interface.py` | test-support env contract | `2b8d026e1bb4688c` |
| `tests/test_h1b.py` | package test (preregistered procedure) | `8bf327d849240e14` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H1b /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h1b.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- lab-original, owner-held, built 2026-10-07

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
