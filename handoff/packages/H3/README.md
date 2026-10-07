# H3 — Recurrent ignition gate + R1 sub-ignition exploratory path

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** REPRODUCED (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

recurrent bistable loop -> hard Heaviside gate on propagation (gate decides ALL propagation); R1 (wired 2026-10-07, K8): when nothing ignites, act on the graded arbitration winner with ZERO consumer propagation — the gate's selectivity is untouched by the exploration path

## Baseline

linear graded propagation (K3 graded control reads linear, error 0.002 — the probe is not blind)

## Key results

- K3: Heaviside step, transition width W 0.100 < 0.12 preregistered (5/5 fresh-seed repro: W 0.08-0.10)
  - Receipt: `flesh-pits receipts repro_k3_ignition_probe.json; package test reproduces W=0.100 standalone`
- K8: R1 recovers frozen phases (phase-2 'c' fraction 0.0->0.79-0.81; total 84-85->128-129) with zero non-feedback consumer deliveries on 19-21 sub-ignition trials per seed
  - Receipt: `receipts/k8_r1_wiring_confirmation.json + k8_r1_wiring_decision.json (vendored in H8; package test reproduces 0.00->0.79/0.81/0.79, 19-21 trials, 0 violations)`
- package test passes K3+R1 standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

K3 linear-probe control; K8 gates a-e (frozen-gains ignition identity 3/3 on behavioral content; K1-K3 byte-identical reruns; K4 non-degradation 4/4; sole-path on sub-ignition trials; freeze recovery replication)

## Generalization bounds

- REPRODUCED 5/5 (K3); R1 confirmed on 3 fresh seeds 51501-51503 (single lab)

## Integration surface

any thresholded propagation point where gate selectivity must survive an exploration requirement

## Expected benefit

genuine ignition event (not weighted averaging) plus freeze-free action selection without weakening the gate — NR-A-004's 'freeze is the price of the gate' is ruled out

## Risks

- theta fixed at 0.45/0.6 (not ADAPTIVE — the R2 adaptive-theta alternative was REJECTED: ~2.7x ignitions, weakened selectivity). Do not tune theta upward expecting free wins

## Rollback

replace the gate with top-k pass-through; revert _select_action to the hold-branch (pre-K8 behavior, byte-identical per post-wire reruns)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/ignition.py` | mechanism | `cc38671f2c0dc6cd` |
| `src/tick.py` | mechanism | `ca78fd8ff9e01d5f` |
| `src/attention.py` | dependency | `ee977a1de7f28200` |
| `src/bids.py` | dependency | `292dc50e7553243a` |
| `src/broadcast.py` | dependency | `8db343efa51728e7` |
| `src/workspace_buffer.py` | dependency | `09eda3384fdf1de2` |
| `src/consumers.py` | dependency | `44096a6d97951b62` |
| `tests/envs.py` | test-support env (R1 only) | `5c6dc33c9d2f43bf` |
| `tests/test_h3.py` | package test (preregistered procedure) | `336a99c56be59076` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H3 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h3.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
