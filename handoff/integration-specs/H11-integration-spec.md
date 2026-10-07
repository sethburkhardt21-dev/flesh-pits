# H11 — MAPIE (D17) donor adoption — integration specification

**Package:** `flesh-pits/handoff/packages/H11` (adoption record + fetch.sh;
donor NOT vendored)
**Maturity:** ADOPT. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is fetched, installed, or
merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

- **Donor:** scikit-learn-contrib/MAPIE — https://github.com/scikit-learn-contrib/MAPIE
- **Pinned SHA:** `3b84b8212db2bba452ef5a09ae06a0dd545869ae` (commit 2026-09-30)
- **License:** BSD-3-Clause (raw LICENSE at SHA, VERIFIED per donor_matrix D17)
- **Primary surface:** uncertainty/confidence outputs on the primary —
  the metacognitive layer. ASSUMED seam from the scratch copy:
  `core/metacognition.py` — `Metacognition.assess_models(...)`
  (prediction_surprise, competence estimates). The scratch copy also
  shows `core/calibration_battery.py`-adjacent lab work, but no live
  calibration module was identified — the primary lane must name the
  actual confidence-output points (Q8).
- **Seam contract:** MAPIE wraps any sklearn-compatible regressor WITHOUT
  modifying it: `SplitConformalRegressor` / `CrossConformalRegressor`
  with `fit` / `conformalize` / `predict_interval` (verified at SHA from
  `mapie/regression/regression.py`). Conformal intervals become the
  calibration GOLD STANDARD: any local uncertainty estimator must report
  empirical coverage vs nominal on a held-out probe before its intervals
  are trusted.

## 2. Interface contract

Fetch (nothing vendored): `./fetch.sh [dest-dir]` clones and checks out
the pinned SHA detached. Dependencies: Python, scikit-learn.

Adapter responsibilities (primary lane):
- **Wrapper, not replacement:** for each primary regressor that emits a
  point prediction with an uncertainty claim (prediction error magnitude,
  competence estimates, retrieval usefulness — wherever the primary
  quantifies uncertainty), wrap it:
  `conformal = SplitConformalRegressor(estimator, confidence_level=0.9)`;
  `conformal.fit(X_calib, y_calib)`; `predict_interval(X_new)` →
  `(y_pred, [y_low, y_high])`. The underlying estimator is untouched —
  rollback is removing the wrapper.
- **Calibration set discipline:** the calibration set must be
  exchangeable with deployment data (the manifest's central bound).
  Hold out a dedicated calibration split; NEVER calibrate on the training
  set. Record the calibration-set hash in the receipt (H7) so coverage
  claims are auditable.
- **What "coverage" means here:** MARGINAL, not conditional — P(y in
  interval) ≥ nominal level averaged over the calibration distribution.
  Intervals can be wide under distribution shift; that width is HONEST,
  not a bug. Do not post-process intervals to look tighter.
- **State ownership:** fitted conformalizers persist with the model
  artifacts they wrap (version them together: estimator-id +
  calibration-hash + MAPIE SHA).

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: a standard calibration mechanism with exact finite-sample
coverage guarantees, sklearn-native, low integration cost. The behavioral
change is metacognitive honesty: uncertainty outputs the primary already
emits become auditable against a gold standard.

Preregistration (stranger-runnable):
1. **Record validation:** `python3 tests/test_h11.py` — validates the
   adoption record standalone. PASS required before fetching.
2. **Fetch integrity:** `./fetch.sh /tmp/mapie-d17`; gate:
   `git rev-parse HEAD` == `3b84b8212db2bba452ef5a09ae06a0dd545869ae`.
3. **Coverage audit (the gold-standard gate):** for each wrapped
   regressor, on a held-out probe: empirical coverage of the 90%
   intervals must be ≥ 0.88 (allowing finite-sample slack; exact nominal
   is 0.90). Gate: pass on the preregistered probe. Any local
   uncertainty estimator whose empirical coverage falls below 0.85 at
   claimed 90% is MISCALIBRATED — its intervals are not trusted until
   recalibrated, and the fact is recorded (this is the calibration
   battery's purpose).
4. **Exchangeability check:** run the coverage audit on a second probe
   drawn from a shifted regime (the primary defines the shift). Gate is
   INFORMATIONAL, not pass/fail: record coverage under shift. If coverage
   collapses, the guarantee does not transfer — label the deployment
   accordingly (manifest bound: non-stationary streams break the
   exchangeability assumption).

## 4. Rollback + tripwire

- **Rollback:** current calibration approach (per manifest) — MAPIE wraps
  regressors without modifying them; remove the wrapper to revert.
- **Tripwire (proves harm):** a scheduled coverage re-audit (test 3's
  probe, frozen). If empirical coverage drops below 0.85 at claimed 90%
  for two consecutive audits, the calibration set has gone stale
  (distribution shift) — quarantine the intervals (mark UNCALIBRATED,
  do not silently widen), recalibrate on fresh data, and record the
  shift. Never "fix" coverage by shrinking intervals.

## 5. Bounds and risks (carried over verbatim, not softened)

- Coverage guarantees assume exchangeability — validate on the primary's
  data regime.
- Marginal (not conditional) coverage; intervals can be wide under
  distribution shift.
- Non-stationary streams break the exchangeability assumption
  (manifest risk).
- Model dependencies: Python; scikit-learn.
