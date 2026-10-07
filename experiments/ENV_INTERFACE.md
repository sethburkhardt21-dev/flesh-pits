# ENV_INTERFACE — Flesh Pits environment/agent contract

**Version:** 1.0.0 (2026-10-07)
**Status:** STABLE — architectures A/B/C/D and all baselines build against this.
Changes require a version bump and a migration note in the registry.

This contract is deliberately minimal. It is the *only* interface between
cognitive architectures (agents) and closed-loop environments. Pure Python,
stdlib only, no third-party dependencies anywhere in the contract.

---

## 1. Environment interface

```python
from env_interface import Environment  # subclass this

class MyEnv(Environment):
    NAME = "my_env"          # str, unique, lowercase_snake
    VERSION = "1.0.0"        # str, bump on any behavior change

    def action_space(self) -> dict: ...
    def observation_space(self) -> dict: ...
    def reset(self, seed: int) -> dict: ...
    def step(self, action: int) -> tuple[dict, float, bool, dict]: ...
    def state_hash(self) -> str: ...
```

### `reset(seed: int) -> obs`

- `seed` is **required** (no implicit global RNG). Same `(NAME, VERSION, seed)`
  and same action sequence ⇒ byte-identical trajectory. This is the
  reproducibility primitive (§46).
- Returns the initial observation dict.

### `step(action: int) -> (obs, reward, done, info)`

- `action` must satisfy `action_space()`; out-of-range actions raise
  `ValueError` (fail closed, never silently clamp).
- `obs`: dict mapping channel name → scalar or tuple-of-scalars.
  JSON-serializable. Values normalized to [-1, 1] or [0, 1] where feasible;
  each environment documents its channels.
- `reward`: float. Each environment documents its reward semantics.
- `done`: bool. True on termination **or** truncation.
- `info`: dict, **metadata only**. Agents are contractually forbidden from
  using `info` for decisions (it may contain ground truth, phase labels,
  debug state). Baselines and architectures are audited against this rule.
  On truncation `info["truncated"] = True`; on true termination it is absent
  or False.

### Spaces

```python
env.action_space()
# {"type": "discrete", "n": 4, "labels": ["north","south","east","west"]}

env.observation_space()
# {"pos_x": {"type": "scalar", "low": 0.0, "high": 1.0, "desc": "..."},
#  "view":  {"type": "vector", "shape": 9, "low": 0.0, "high": 1.0, "desc": "..."},
#  "cue":   {"type": "categorical", "values": [0, 1, 2], "desc": "..."}}
```

Space types: `scalar` (low/high), `vector` (shape, low/high),
`categorical` (values list). `discrete` actions only in v1.

### `state_hash() -> str`

SHA-256 hex over the canonical internal state (layout, positions, RNG
state, step counters). Used for §46 reproducibility receipts:
initial hash after `reset`, final hash at episode end.

---

## 2. Agent interface

```python
from env_interface import Agent  # subclass this

class MyAgent(Agent):
    NAME = "my_agent"

    def reset(self, seed: int, action_space: dict) -> None: ...
    def act(self, obs: dict) -> int: ...
    def update(self, obs, action, reward, done, info) -> None: ...
    def snapshot(self) -> dict: ...     # serializable internal state
    def restore(self, state: dict) -> None: ...
```

- `act` is pure w.r.t. the agent's own state: it may read agent memory,
  never the environment.
- Learning happens in `update` (post-step). Agents with no learning
  implement it as a no-op — explicitly, not by omission.
- `reset()` reinitializes PER-EPISODE state (RNG, counters). Cross-episode
  learning persists in the agent OBJECT: the harness reuses one agent
  instance across all episodes of a run, calling `reset()` each episode.
  An agent that wipes learned parameters in `reset()` is a no-learning
  agent by construction — document which yours is.
- `snapshot`/`restore` enable continuity tests (§29: restart, crash,
  checkpoint restore). State must be JSON-serializable.
- Agents receive `info` in `update` but **must not** use it for decisions
  (same rule as environments' info). Document any exception.

---

## 3. Episode / run ID scheme

```
episode_id = f"{env_name}:v{env_version}:seed={seed}:run={run_index}"
```

- `run_index` counts episodes within one experiment run (0-based).
- The harness derives per-episode seeds deterministically:
  `episode_seed = derive_seed(primary_seed, run_index, "env")`.
  Primary seed + run index fully determine the experiment.

---

## 4. Harness loop (canonical)

```python
obs = env.reset(seed=episode_seed)
agent.reset(seed=agent_seed, action_space=env.action_space())
done = False
while not done:
    action = agent.act(obs)
    obs2, reward, done, info = env.step(action)
    agent.update(obs, action, reward, done, info)
    obs = obs2
```

`agent_seed` is derived from the episode seed (`episode_seed ^ 0x9E3779B9`)
so agent stochasticity is reproducible without sharing the env RNG stream.

---

## 5. Conventions

- **Determinism:** no `random` module globals, no `time`, no wall-clock
  inside envs or agents. All randomness from `random.Random(seed)` instances.
- **Neutrality:** observations contain no identity tokens, no privileged
  strings, no affective labels. Arbitrary entities are symmetric integers.
  (Directive §14–15.)
- **Stdlib only:** `random`, `hashlib`, `json`, `math`, `dataclasses`,
  `abc`, `collections`, `itertools`. No numpy, no torch, no requests.
- **Lineage:** every env file carries `NAME` and `VERSION`; every receipt
  records them plus the config hash.

---

## 6. Directory layout (this tree)

```
flesh-pits/experiments/
  ENV_INTERFACE.md        # this contract
  env_interface.py        # ABCs + helpers (episode IDs, seed derivation)
  envs/                   # the §35 environments, one file each
  baselines/              # the §39 baselines, one file each + agent_base.py
  harness.py              # run_episode / run_experiment / receipts
  tests/                  # unittest suites (stdlib)
  EXPERIMENTS_REGISTRY.md # append-only §38 log  (note: filename per task)
flesh-pits/prototypes/
  arch_d.py               # Architecture D, the lean causal core
flesh-pits/receipts/      # per-run JSON receipts
flesh-pits/research/
  negative_results.md     # preserved failures (§40, §52)
```

Note: the registry file is `EXPERIMENT_REGISTRY.md` (task §38 names it
`experiments/EXPERIMENT_REGISTRY`; kept at that exact name).
