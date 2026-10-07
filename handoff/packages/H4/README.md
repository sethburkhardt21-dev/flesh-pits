# H4 — change_bid (RunningZScoreBid)

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** CAUSAL (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

sigmoid of causal running z-score — change detection + habituation in one mechanism; fail-closed on non-finite input

## Baseline

raw thresholds

## Key results

- CAUSAL in two independent tests (battery: zero-latency shift detection, drift 22 vs raw 69; A: K1/K4 loop)
  - Receipt: `Phase-4 battery receipts`
- K7: bids lesion collapses arbitration winner-entropy 1.03-1.38 -> 0.0 bits
  - Receipt: `lab K7 receipt; package test reproduces the collapse 1.00 -> 0.00 bits on an alternation probe`
- package test passes shift/habituation/fail-closed/entropy standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

K7 IS the ablation (experiment-side constant-0.5 stubs)

## Generalization bounds

- CAUSAL for moment-to-moment arbitration under frozen gains; NR-A-009 bound — the learned system barely needs bids (learned gains compensate). Do not claim bids drive the K4 reward win

## Integration surface

any novelty/salience input needing a cheap honest signal

## Expected benefit

cheap honest novelty signal with receipts; runs both standalone and inside the attention loop

## Risks

- none identified beyond the NR-A-009 bound

## Rollback

trivial — raw thresholds

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/bids.py` | mechanism | `292dc50e7553243a` |
| `tests/test_h4.py` | package test (preregistered procedure) | `3e141bdc1c7a4ef7` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H4 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h4.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- audit twin at emergent-mind/sandbox/being/self_model_port/change_bid.py (same mechanism; not vendored — outside flesh-pits)

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
