# H11 — MAPIE (D17) — donor adoption

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** ADOPT (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** BSD-3-Clause (raw LICENSE at SHA, VERIFIED per donor_matrix).
**Model dependencies:** Python; scikit-learn.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

conformal prediction with finite-sample coverage guarantees: SplitConformalRegressor / CrossConformalRegressor, fit/conformalize/predict_interval (verified at SHA from mapie/regression/regression.py)

## Baseline

the primary's current calibration approach

## Key results

- donor_matrix D17: ADOPT — SplitConformalRegressor/CrossConformalRegressor with fit/conformalize/predict_interval confirmed at SHA
  - Receipt: `emergent-mind/research/donor_matrix.md (D17 entry)`
- package test validates the adoption record standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

n/a (donor)

## Generalization bounds

- coverage guarantees assume exchangeability — validate on the primary's data regime
- marginal (not conditional) coverage; intervals can be wide under distribution shift

## Integration surface

uncertainty/confidence outputs on the primary (metacognitive layer)

## Expected benefit

standard calibration mechanism: exact coverage guarantees, sklearn-native, low integration cost

## Risks

- non-stationary streams break the exchangeability assumption

## Rollback

current calibration approach (MAPIE wraps regressors without modifying them)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `ADOPTION.md` | adoption record | `14b2c93f54a61545` |
| `fetch.sh` | fetch script (pins SHA) | `8eec8c0891b55bc7` |
| `tests/test_h11.py` | package test (preregistered procedure) | `36e17c33107babc8` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H11 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h11.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- donor NOT vendored; fetch.sh pins the exact SHA
- pinned SHA 3b84b8212db2bba452ef5a09ae06a0dd545869ae (2026-09-30)

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
