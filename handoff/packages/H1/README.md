# H1 — A's learned-gain attention loop (arbitration + learned gains)

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** GENERALIZING (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

habituated z-score bids x learned per-channel gains, argmax with label-agnostic tie-break; delta-rule gain update on observed channels only, capped, from broadcast feedback (attention_update consumer closes the loop)

## Baseline

frozen gains (fixed baseline) on the canonical changing_rule env (ENV_INTERFACE v1.0.0)

## Key results

- K4 learned/frozen total-reward ratios 1.61/2.03/1.95/1.57 (4/4 seeds, preregistered margin 1.30)
  - Receipt: `flesh-pits/receipts/K4-CANONICAL-RERUN.ndjson (hash-chained); vendored copy in receipts/`
- multi-seed repro 1.71-2.13 on 5/5 fresh seeds
  - Receipt: `flesh-pits receipts repro_k4_attention_baseline.json`
- package test reproduces 1.61/2.03/1.95/1.57 exactly, standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

freezing gains collapses the advantage (K4 IS the ablation); K7 bids lesion collapses arbitration entropy 1.03-1.38 -> 0.0 bits but learned gains compensate for dead bids (NR-A-009 — the win is carried by the gain loop, not the bids)

## Generalization bounds

- GENERALIZING: noisy-signal tracking (K5 P2: 4/4, R 1.31-1.52); multi-reversal stationary shifts (K5 P3: 4/4, R 1.74-2.05)
- NR-A-005: frozen change-bids suffice on signal-tracking reversals
- NR-A-006 (STRUCTURAL): delayed_reward — learned LOSES, R 0.08-0.17; K9 redesign failed 0/4 (NR-A-011). Do not deploy on sparse-delayed-reward tasks
- NR-A-007-without-cue-input: cue-conditioned changing_rule without cue input R 1.03-1.09 (architectural input bound; LIFTED by H1b when the cue is in the input)

## Integration surface

any selection/competition point on the primary that needs adaptive prioritization with receipts

## Expected benefit

adaptive attention that provably tracks salience-orthogonal relevance shifts; identity-symmetric (bit-identical trajectories for novel labels)

## Risks

- NR-A-006 bound travels with it — never deploy on sparse-delayed-reward tasks (K9 redesign failed; structural)
- without the cue-indexed adapter (H1b), do not deploy on cue-conditioned tasks

## Rollback

freeze gains (the K4 frozen mode is a one-flag rollback to the fixed baseline)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/attention.py` | mechanism | `ee977a1de7f28200` |
| `src/bids.py` | mechanism | `292dc50e7553243a` |
| `src/tick.py` | mechanism | `ca78fd8ff9e01d5f` |
| `src/consumers.py` | mechanism | `44096a6d97951b62` |
| `src/broadcast.py` | dependency | `8db343efa51728e7` |
| `src/workspace_buffer.py` | dependency | `09eda3384fdf1de2` |
| `src/ignition.py` | dependency | `cc38671f2c0dc6cd` |
| `tests/envs.py` | test-support env (not part of the mechanism) | `5c6dc33c9d2f43bf` |
| `tests/test_h1.py` | package test (preregistered procedure) | `e3c56bb7c0dba070` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H1 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h1.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- Phase-4 twin at brothel/prototypes/architecture-a/ (not vendored)
- maturity audit 2026-10-07T07:30:37Z GREEN; K8 wire receipt k8_r1_wiring_decision.json

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
