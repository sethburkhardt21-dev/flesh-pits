# H8 — K8 wire-by-decision-receipt discipline (method, not a mechanism)

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** EXECUTED (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

five preregistered gates (behavioral identity, byte-identical reruns, non-degradation, sole-path, recovery replication) + post-wire byte-identical reruns of every prior experiment, all in one hash-chained decision receipt

## Baseline

ad-hoc wiring

## Key results

- WIRE decision recorded 2026-10-07T07:23:05Z; all gates PASS on fresh seeds; post-wire K1-K4 reruns byte-identical (sha prefixes e007671877da7e95, bd2260363c15c, 0fc80361213e7c93, 4832d984137bfb6d)
  - Receipt: `receipts/k8_r1_wiring_decision.json + receipts/k8_r1_wiring_confirmation.json (vendored)`
- package test demonstrates the five-gate procedure on a toy wiring standalone, including the live REJECT null
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

n/a (method); the gates ARE the ablation of silent behavior change

## Generalization bounds

- the method is only as good as the gates — behavioral-identity must name WHAT is compared (NR-A-010 lesson: behavioral content, not process-global sequence numbers)

## Integration surface

any future wiring of lab findings into a tick path — primary or lab

## Expected benefit

wiring that cannot silently change behavior; the null (REJECT) was live and preregistered

## Risks

- gate quality is the method's ceiling

## Rollback

n/a (method)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/k8_wiring_method_reference.py` | method reference (the actual K8 script, renamed) | `892e06bf8c82e18a` |
| `tests/test_h8.py` | package test (preregistered procedure) | `30ee61eed23b79fa` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H8 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h8.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- package test uses a toy wiring to demonstrate the procedure; canonical evidence is the vendored K8 decision receipt

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
