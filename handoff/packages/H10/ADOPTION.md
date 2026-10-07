# H10 — cleanrl (D9) adoption record

- **Repo / URL:** vwxyzjn/cleanrl — https://github.com/vwxyzjn/cleanrl
- **Pinned SHA:** `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` (commit 2026-04-20)
- **License:** MIT (raw LICENSE at SHA, VERIFIED per donor_matrix D9)
- **License note:** permissive; attribution only.

## Mechanism
Single-file, readable RL implementations — PPO, DQN, C51, SAC (PyTorch
and JAX variants); curiosity via **RND** (`cleanrl/ppo_rnd_envpool.py`).
NOTE (verified at SHA): there is NO `rnd_atari.py` at this SHA — the
RND/curiosity entry point is `cleanrl/ppo_rnd_envpool.py`.

## Vendored artifact
`src/ppo_rnd_envpool.py` — the single-file RND/curiosity reference,
fetched 2026-10-07 from raw.githubusercontent.com at the pinned SHA.
- sha256: `d6ec2fb0f2f1e3f78624b001f02abbe5fc46a645e2f88b2ff644609e3d1e5560`
- Compiles clean under py_compile; defines `RNDModel` (line 184) and the
  intrinsic-reward coefficient path (AST-verified by tests/test_h10.py).
- Dependencies to RUN it (not vendored): Python, PyTorch, envpool, gym,
  numpy, tyro.

## Disposition (donor_matrix D9): ADOPT
Adopt `ppo_rnd_envpool.py` as the standard RND/curiosity reference for
intrinsic-motivation experiments; single-file PPO/DQN as disposable RL
baselines.

## Integration surface
Intrinsic-motivation experiments on the primary: the RND prediction-error
bonus is the canonical curiosity baseline any local intrinsic-reward
mechanism must beat.

## Benchmark plan
RND intrinsic-motivation bonus vs the primary's local intrinsic-reward
mechanism on a sparse-reward probe (e.g. the lab's delayed_reward env):
compare exploration coverage and time-to-first-reward.

## Risks
Research-grade single file, not a library; algorithm internals not
line-verified individually at this SHA (single-file form makes
verification cheap — do it before trusting a specific line).
Env-dependent for full runs (envpool/gym).

## Rollback
Drop the baseline. The file is a reference, not a dependency — nothing
in the primary imports it unless an experiment explicitly does.
