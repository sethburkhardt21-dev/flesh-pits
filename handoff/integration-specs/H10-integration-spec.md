# H10 — cleanrl ppo_rnd_envpool.py (D9) donor adoption — integration specification

**Package:** `flesh-pits/handoff/packages/H10` (vendored donor file +
adoption record)
**Maturity:** ADOPT. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is imported or merged into
any primary tree.

## 1. Integration surface

- **Donor:** vwxyzjn/cleanrl — https://github.com/vwxyzjn/cleanrl
- **Pinned SHA:** `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` (commit 2026-04-20)
- **Vendored artifact:** `src/ppo_rnd_envpool.py` (22,283 bytes; sha256
  `d6ec2fb0f2f1e3f78624b001f02abbe5fc46a645e2f88b2ff644609e3d1e5560`,
  fetched 2026-10-07)
- **License:** MIT (raw LICENSE at SHA, VERIFIED per donor_matrix D9)
- **Primary surface:** intrinsic-motivation EXPERIMENTS on the primary.
  This is a **reference baseline, not a dependency** — nothing in the
  primary imports it unless an experiment explicitly does (manifest
  rollback: "drop the baseline"). The natural experiment partner is the
  H7 harness: RND runs as one agent arm under the H7 env/agent contract.
- **Verified at SHA:** there is NO `rnd_atari.py` at this SHA — the
  RND/curiosity entry point is `cleanrl/ppo_rnd_envpool.py`. Do not go
  looking for the other file.

## 2. Interface contract

Mechanism (single-file): `RNDModel` predictor–target pair (line 184),
intrinsic-reward coefficient wiring into PPO (envpool). Package test
AST-verifies `RNDModel` + the intrinsic path and hash-matches the file
(compiles clean under py_compile).

Adapter responsibilities (primary lane):
- **Reference, not import:** the file lives in the primary's experiment
  area (or stays in the lab) as the canonical curiosity baseline. Any
  local intrinsic-reward mechanism the primary builds must BEAT this
  baseline on a preregistered probe — that is the integration's entire
  behavioral purpose.
- **Dependencies to RUN (not vendored):** Python, PyTorch, envpool, gym,
  numpy, tyro. These are heavyweight — run RND experiments in the
  experiment environment, never in the live tick path. RND is an
  experiment-harness citizen (H7), not a tick module.
- **Portability note:** envpool/gym may not exist in the primary's
  runtime. The fallback is to re-implement the RND bonus
  (predictor–target MSE on observations) against the H7 `Agent` contract
  — the algorithm is the baseline, not the file. If re-implemented,
  the re-implementation must reproduce the file's intrinsic-reward
  computation on a fixed observation tape (byte-identical bonuses)
  before it counts as "the RND baseline."
- **Error behavior:** none specified by the donor (research-grade single
  file). The primary's experiment wrapper owns failure handling.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: a canonical RND/curiosity baseline exists that any local
intrinsic-reward mechanism must beat; disposable single-file PPO/DQN
baselines available. No cognitive claim — this is measurement
infrastructure.

Preregistration (stranger-runnable):
1. **Artifact validation:** `python3 tests/test_h10.py` in the package —
   file compiles, AST-verifies `RNDModel` + intrinsic path, sha256
   matches. PASS required.
2. **Line verification (manifest bound):** algorithm internals are NOT
   line-verified individually at this SHA — before trusting a specific
   line (e.g. the intrinsic coefficient schedule), read it. Gate: the
   primary lane records which lines the experiment depends on and has
   read them.
3. **Baseline race:** RND intrinsic bonus vs the primary's local
   intrinsic-reward mechanism on a sparse-reward probe (the lab's
   `delayed_reward` env is the reference probe): compare exploration
   coverage and time-to-first-reward, preregistered seeds. Gate for
   adopting the local mechanism: beats RND on both metrics, 3/3 seeds.
   (Note the H1 NR-A-006 bound: learned attention LOSES on
   delayed_reward — intrinsic motivation is the open problem there, not
   a solved one. Do not expect RND to win trivially either.)

## 4. Rollback + tripwire

- **Rollback:** drop the baseline (per manifest) — reference only;
  nothing imports it unless an experiment explicitly does. Delete the
  file and the experiments that used it keep their receipts (H7 chain
  is self-verifying).
- **Tripwire:** if an experiment's results depend on RND behavior at a
  line that was never read (violating test 2's gate), mark the result
  UNVERIFIED pending line review. If torch/envpool versions drift,
  re-run the artifact validation — the pin is on the cleanrl SHA, not
  on the dependency tree.

## 5. Bounds and risks (carried over verbatim, not softened)

- Research-grade single file, not a library.
- Algorithm internals not line-verified individually at this SHA
  (single-file form makes verification cheap — do it before trusting a
  specific line).
- Env-dependent for full runs (envpool/gym).
- Model dependencies: Python; PyTorch; envpool/gym (to RUN; not to vendor).
