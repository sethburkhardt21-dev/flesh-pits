# H11 — MAPIE (D17) adoption record

- **Repo / URL:** scikit-learn-contrib/MAPIE — https://github.com/scikit-learn-contrib/MAPIE
- **Pinned SHA:** `3b84b8212db2bba452ef5a09ae06a0dd545869ae` (commit 2026-09-30)
- **License:** BSD-3-Clause (raw LICENSE at SHA, VERIFIED per donor_matrix D17)
- **License note:** permissive; attribution only.

## Mechanism
**Conformal prediction** with finite-sample coverage guarantees:
SplitConformalRegressor, CrossConformalRegressor —
`fit`/`conformalize`/`predict_interval` (verified at SHA from
`mapie/regression/regression.py`, per donor_matrix D17).
Distribution-free prediction intervals with exact marginal coverage for
any sklearn regressor.

## Disposition (donor_matrix D17): ADOPT
Adopt conformal prediction (MAPIE) as the standard **calibration**
mechanism for the primary's metacognitive layer: exact coverage
guarantees, sklearn-native, low integration cost, maintained (Sep 2026).
The natural scoring partner to uncertainty-toolbox metrics.

## Integration surface
Uncertainty/confidence outputs on the primary — wraps any
sklearn-compatible regressor; conformal intervals become the calibration
gold standard for local uncertainty mechanisms.

## Benchmark plan
Coverage/width of conformal intervals as the calibration gold standard:
any local uncertainty estimator must report empirical coverage vs
nominal on a held-out probe before its intervals are trusted.

## Risks
Coverage guarantees assume **exchangeability** — validate on the
primary's data regime (non-stationary streams break the guarantee).
Marginal (not conditional) coverage; intervals can be wide under
distribution shift.

## Rollback
Current calibration approach. MAPIE wraps regressors without modifying
them — remove the wrapper to revert.

## Fetch
`./fetch.sh` clones the repo and checks out the pinned SHA (detached HEAD).
Nothing is vendored in this package — the donor is fetched at adoption time
from the pinned commit.
