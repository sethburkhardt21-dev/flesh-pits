# H2 — Bounded workspace buffer + sole-path broadcast pattern

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** REPRODUCED (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** lab-original, owner-held.
**Model dependencies:** none — float arithmetic only; no foundation-model call in the loop.

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

capacity-K bounded buffer with explicit eviction policy (newcomer competes unprotected, TTL decay, per-admission receipts); in-memory bus as the ONLY consumer path (declared consumers, per-item consumer sets, lesion/restore, delivery receipts)

## Baseline

unbounded queue / unwired broadcast (Delta=0 theater — Phase-4 battery 001b bit-identical trajectories)

## Key results

- K1: interference index 0/0.507/1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf
  - Receipt: `lab K1 receipt; package test reproduces 0/0.507/1.0 exactly`
- K2: total lesion silences all six consumers at once, internal processing continues, full recovery, no side channels (Phase-4 coordinator re-ran K2 by its own hand — PASS, numbers match)
  - Receipt: `lab K2 receipt`
- REPRODUCED 5/5 fresh seeds on the original code paths
  - Receipt: `flesh-pits receipts repro_k1_capacity_lesion.json, repro_k2_broadcast_lesion.json`
- package test passes K1+K2 standalone
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

K1 (capacity lesion) and K2 (broadcast lesion) ARE the ablations

## Generalization bounds

- REPRODUCED on 5/5 fresh seeds; single lab — no independent replication yet (stated, not hidden)

## Integration surface

any multi-consumer integration point on the primary; the K2 sole-path test is the acceptance gate

## Expected benefit

provable global availability — a broadcast that demonstrably changes downstream consumers (the lab's anti-theater law: broadcast without measured consumer change is theater)

## Risks

- adds a bottleneck; the burden of proof (K1/K2) must travel with the pattern — re-run K2 against the primary's consumer set or drop the claim

## Rollback

remove the bus, direct calls (the pre-bus wiring is the trivial fallback)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/workspace_buffer.py` | mechanism | `09eda3384fdf1de2` |
| `src/broadcast.py` | mechanism | `8db343efa51728e7` |
| `src/tick.py` | mechanism | `ca78fd8ff9e01d5f` |
| `src/consumers.py` | mechanism | `44096a6d97951b62` |
| `src/attention.py` | dependency | `ee977a1de7f28200` |
| `src/bids.py` | dependency | `292dc50e7553243a` |
| `src/ignition.py` | dependency | `cc38671f2c0dc6cd` |
| `tests/envs.py` | test-support env (K2 only) | `5c6dc33c9d2f43bf` |
| `tests/test_h2.py` | package test (preregistered procedure) | `f0c08687bc44f772` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H2 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h2.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
