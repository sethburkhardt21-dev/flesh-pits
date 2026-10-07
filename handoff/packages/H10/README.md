# H10 — cleanrl ppo_rnd_envpool.py (D9) — donor adoption

**Harvest class:** HARVEST_NOW (directive §49). **Maturity:** ADOPT (flesh-pits/bin/maturity_audit_report.json, generated 2026-10-07T12:14:34Z).
**License:** MIT (raw LICENSE at SHA, VERIFIED per donor_matrix).
**Model dependencies:** Python; PyTorch; envpool/gym (to RUN; not to vendor).

**Status: PACKAGED ONLY.** Staged for the primary effort's decision per §51 reorient. Nothing here is merged into primary, and nothing here may be auto-merged.

## Mechanism

single-file RND/curiosity intrinsic-motivation reference: RNDModel predictor-target pair, intrinsic-reward coefficient wiring into PPO (envpool)

## Baseline

no intrinsic-motivation baseline

## Key results

- donor_matrix D9: ADOPT — single-file layout confirmed at SHA; RND entry point is ppo_rnd_envpool.py (no rnd_atari.py at this SHA)
  - Receipt: `emergent-mind/research/donor_matrix.md (D9 entry)`
- package test: file compiles, AST-verifies RNDModel + intrinsic path, sha256 matches
  - Receipt: `receipts/package_test_receipt.json (generated on test run)`

## Ablation

n/a (donor)

## Generalization bounds

- algorithm internals not line-verified individually at this SHA (single-file form makes verification cheap — do it before trusting a specific line)
- env-dependent for full runs

## Integration surface

intrinsic-motivation experiments on the primary

## Expected benefit

canonical RND/curiosity baseline any local intrinsic-reward mechanism must beat; disposable single-file PPO/DQN baselines

## Risks

- research-grade single file, not a library

## Rollback

drop the baseline (reference only; nothing imports it unless an experiment explicitly does)

## Package contents

| File | Role | sha256 (prefix) |
|---|---|---|
| `src/ppo_rnd_envpool.py` | vendored donor file (MIT) | `d6ec2fb0f2f1e3f7` |
| `ADOPTION.md` | adoption record | `0506c254892743e6` |
| `tests/test_h10.py` | package test (preregistered procedure) | `207548b90be096c8` |

## Running the package test

Copy the package to a scratch dir and run the test with the package root as the working directory; it must pass with no access to the lab tree:

```sh
cp -r H10 /tmp/pkgtest && cd /tmp/pkgtest
python3 tests/test_h10.py
```

Recorded result: **PASS standalone (copied to /tmp, ran without the lab tree)**.

## Notes

- vendored file sha256 d6ec2fb0f2f1e3f78624b001f02abbe5fc46a645e2f88b2ff644609e3d1e5560, fetched 2026-10-07
- pinned SHA fe8d8a03c41a7ef5b523e2e354bd01c363e786bb (2026-04-20)

---

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7, §51 reorient). Lab tree left intact; sources copied, not moved. CONSCIOUSNESS: UNRESOLVED.*
