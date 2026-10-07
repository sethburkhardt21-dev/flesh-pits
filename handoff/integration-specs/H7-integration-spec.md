# H7 — experiment harness (preregistration + hash-chained receipts) — integration specification

**Package:** `flesh-pits/handoff/packages/H7` (src sha256 in `manifest.json`)
**Maturity:** EXECUTED. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface

This is **metrology, not a cognitive mechanism** — it does not touch the
tick, the workspace, or memory. The integration surface is the primary's
**experiment/verification workflow**: any future primary-side experiment
adopts the harness as its procedure standard.

- **Primary surface:** wherever the primary runs evaluations, ablations,
  or verification probes (no specific module is assumed; the harness is
  standalone). If the primary has an existing eval/verification runner,
  the decision is adopt-vs-converge (see §2, consolidation note).
- **Boundary (stated in the package):** the lab's env/agent zoo
  (`envs`, `baselines`, `arch_d`) is NOT vendored — the package proves
  the metrology machinery with stubbed envs. The primary supplies its own
  env/agent implementations against the `env_interface.py` contract.

## 2. Interface contract

Vendored: `src/harness.py`, `src/env_interface.py`.

```python
# env_interface.py — the contract any env/agent must satisfy
class Environment(ABC):
    NAME: str; VERSION: str
    def reset(self, seed: int) -> dict
    def step(self, action: int) -> (obs, reward: float, done: bool, info: dict)
    def action_space(self) -> dict; def observation_space(self) -> dict
    def state_hash(self) -> str
    def validate_action(self, action) -> int
    @staticmethod
    def validate_obs(obs, space) -> dict
class Agent(ABC):
    def reset(self, seed: int, action_space: dict) -> None
    def act(self, obs: dict) -> int
    def update(self, obs, action, reward, done, info) -> None
    def snapshot(self) -> dict; def restore(self, state: dict) -> None
def make_episode_id(env_name, env_version, seed, run_index) -> str
def derive_seed(primary_seed, run_index, stream: str) -> int   # stream-separated RNG
def canonical_json(obj) -> str; def config_hash(config) -> dict

# harness.py — the experiment procedure
def run_episode(env, agent, episode_seed, run_index, max_steps=None, validate=False) -> dict
    # returns {episode_id, seed, init_hash, final_hash, return, steps, done, truncated}
def run_experiment(env_name, agent_name, n_episodes, primary_seed,
                   experiment_id, max_steps=None, validate=False) -> dict
    # agent learns ACROSS episodes; env phases advance; returns full result dict
    # with config, config_hash, primary_seed, started_utc, episodes[], summary{}
def write_receipt(result, receipts_dir, hypothesis="", null="",
                  preregistered_metric="", baseline="", conditions="",
                  interpretation="", limitations="") -> str  # path
    # hash-chained: prev_receipt_hash links to the previous chained receipt;
    # receipt_hash = sha256 of the body. FAILS CLOSED per the lab's receipt
    # discipline (this spec's task instructions carry the same rule).
def verify_chain(receipts_dir) -> (ok: bool, problems: list)
```

Every serious experiment run under this harness carries: hypothesis,
null hypothesis, preregistered metric, baseline, ablation, procedure,
seed/config, result, interpretation, limitations (§38 standard). Seeds are
stream-separated (`derive_seed` with "env"/"agent" streams) so env and
agent RNG never share a stream.

**Consolidation note (risk, carried over):** two harness variants exist
(flesh-pits + brothel). Before primary adoption, consolidate to ONE —
diff the brothel twin against this one, keep the union of guarantees
(preregistration schema + chained receipts + stream-separated seeds are
non-negotiable), and record which variant's behaviors were dropped.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: every future claim arrives with a receipt. Proven on the
full K-series (9+8 receipts Phase 4; K-series + repro + K8 receipts
Phase 5); caught NR-A-008 (mis-specified metric) and NR-A-010 (harness
artifact) honestly — the harness catches its own artifacts, which is the
point.

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H7 /tmp/pkgtest && python3
   tests/test_h7.py` — episode loop, preregistration schema, chained
   receipts, tamper evidence, PASS.
2. **Chain-integrity drill on the primary's receipts dir:** write three
   receipts, tamper with the middle file's bytes, run `verify_chain` —
   gate: `(False, problems)` naming the tampered file; untampered prefix
   still verifies.
3. **Env/agent conformance:** implement one primary env and one primary
   agent against `env_interface.py`; run with `validate=True` — gate: no
   validation errors and `state_hash` stable across identical seeds
   (determinism check).
4. **Preregistration completeness:** any primary experiment claiming a
   result must have its preregistration fields (hypothesis, null, metric,
   baseline, ablation, seeds) present in the receipt BEFORE the run —
   gate: a receipt linter (10 lines) rejects metric-after-result.

## 4. Rollback + tripwire

- **Rollback:** n/a (metrology — per manifest). Adoption is additive:
  stop using the harness and the old ad-hoc scripts remain. Receipts
  already written stay valid (the chain is self-verifying).
- **Tripwire (proves harm):** `verify_chain` runs on every receipts-dir
  write in CI. A broken chain is a fail-closed event: no further claims
  are published from that directory until the break is explained. If
  `derive_seed` stream separation is ever collapsed (env and agent sharing
  a stream), all experiment IDs from the affected period are marked
  CONTAMINATED — this is the NR-A-010 class of artifact.

## 5. Bounds and risks (carried over verbatim, not softened)

- Two harness variants exist (flesh-pits + brothel) — consolidate to one
  before primary adoption.
- The lab's env/agent zoo is stubbed, not vendored — the primary must
  supply real envs/agents; the harness does not test cognition by itself.
- EXECUTED maturity: the harness is proven as metrology, not as a
  cognitive result. Do not cite harness adoption as evidence for any
  mechanism claim.
