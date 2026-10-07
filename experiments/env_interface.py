"""ENV_INTERFACE executable contract — Flesh Pits closed-loop lab.

Pure Python, stdlib only. Every environment subclasses ``Environment``;
every cognitive architecture / baseline subclasses ``Agent``.

Contract version: 1.0.0
"""

import hashlib
import json
import random
from abc import ABC, abstractmethod

CONTRACT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_episode_id(env_name: str, env_version: str, seed: int, run_index: int) -> str:
    """Canonical episode/run ID scheme.

    episode_id = "{env_name}:v{env_version}:seed={seed}:run={run_index}"
    """
    return f"{env_name}:v{env_version}:seed={seed}:run={run_index}"


def derive_seed(primary_seed: int, run_index: int, stream: str = "env") -> int:
    """Deterministically derive a per-episode/per-stream seed.

    Separate streams ("env", "agent") keep environment stochasticity and agent
    stochasticity independent while remaining fully reproducible.
    """
    h = hashlib.sha256(f"{primary_seed}:{run_index}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")  # 31-bit positive int


def canonical_json(obj) -> str:
    """Canonical JSON encoding (sorted keys, compact separators) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict) -> str:
    """SHA-256 over the canonical JSON of an experiment configuration."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def new_rng(seed: int) -> random.Random:
    """The only sanctioned source of randomness: an isolated Random instance."""
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Environment ABC
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Closed-loop environment. Subclass and implement all abstract members."""

    NAME: str = "base_env"
    VERSION: str = "0.0.0"

    def __init__(self):
        self._rng = new_rng(0)
        self._steps = 0

    # -- spaces ------------------------------------------------------------
    @abstractmethod
    def action_space(self) -> dict:
        """E.g. {"type": "discrete", "n": 4, "labels": [...]}."""

    @abstractmethod
    def observation_space(self) -> dict:
        """Field -> {"type": "scalar"|"vector"|"categorical", ...} spec."""

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def reset(self, seed: int) -> dict:
        """Reset to initial state using the given seed. Returns initial obs."""

    @abstractmethod
    def step(self, action: int):
        """Apply action. Returns (obs, reward, done, info)."""

    # -- introspection -----------------------------------------------------
    def state_hash(self) -> str:
        """SHA-256 hex of canonical internal state. Override for real state."""
        return hashlib.sha256(
            canonical_json({"steps": self._steps}).encode("utf-8")
        ).hexdigest()

    # -- validation --------------------------------------------------------
    def validate_action(self, action) -> int:
        space = self.action_space()
        if space.get("type") != "discrete":
            raise NotImplementedError("v1 contract supports discrete actions only")
        n = space["n"]
        if not isinstance(action, int) or isinstance(action, bool):
            raise ValueError(f"action must be int in [0, {n}), got {action!r}")
        if not (0 <= action < n):
            raise ValueError(f"action {action} out of range [0, {n})")
        return action

    @staticmethod
    def validate_obs(obs: dict, space: dict) -> dict:
        """Check an observation dict against its declared space. Fail closed."""
        for field, spec in space.items():
            if field not in obs:
                raise ValueError(f"obs missing declared field {field!r}")
            v = obs[field]
            t = spec["type"]
            if t == "scalar":
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(f"field {field!r} must be numeric, got {v!r}")
                if not (spec["low"] - 1e-9 <= v <= spec["high"] + 1e-9):
                    raise ValueError(f"field {field!r}={v} outside [{spec['low']}, {spec['high']}]")
            elif t == "vector":
                if not isinstance(v, (list, tuple)) or len(v) != spec["shape"]:
                    raise ValueError(f"field {field!r} must be vector of len {spec['shape']}")
                for x in v:
                    if not isinstance(x, (int, float)) or isinstance(x, bool):
                        raise ValueError(f"field {field!r} contains non-numeric {x!r}")
            elif t == "categorical":
                if v not in spec["values"]:
                    raise ValueError(f"field {field!r}={v!r} not in {spec['values']}")
            else:
                raise ValueError(f"unknown space type {t!r}")
        return obs


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------

class Agent(ABC):
    """Cognitive architecture / baseline. Subclass and implement all members."""

    NAME: str = "base_agent"

    @abstractmethod
    def reset(self, seed: int, action_space: dict) -> None:
        """(Re)initialize internal state for a new episode."""

    @abstractmethod
    def act(self, obs: dict) -> int:
        """Select an action from the observation. Must not touch the env."""

    @abstractmethod
    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        """Post-step hook. Learning happens here; no-op if the agent is static.

        NOTE: `info` is metadata only and MUST NOT drive decisions.
        """

    @abstractmethod
    def snapshot(self) -> dict:
        """Return JSON-serializable internal state (continuity / restart tests)."""

    @abstractmethod
    def restore(self, state: dict) -> None:
        """Restore internal state from a snapshot()."""


# ---------------------------------------------------------------------------
# Shared observation utilities (used by agents; pure functions of obs)
# ---------------------------------------------------------------------------

def obs_to_vector(obs: dict, space: dict) -> list:
    """Flatten an observation dict to a float vector in declared field order.

    categorical fields are expanded to one-hot. Deterministic given (obs, space).
    """
    vec = []
    for field, spec in space.items():
        v = obs[field]
        t = spec["type"]
        if t in ("scalar",):
            vec.append(float(v))
        elif t == "vector":
            vec.extend(float(x) for x in v)
        elif t == "categorical":
            for val in spec["values"]:
                vec.append(1.0 if v == val else 0.0)
    return vec


def vec_dim(space: dict) -> int:
    """Dimension of obs_to_vector output for a space."""
    d = 0
    for spec in space.values():
        t = spec["type"]
        if t == "scalar":
            d += 1
        elif t == "vector":
            d += spec["shape"]
        elif t == "categorical":
            d += len(spec["values"])
    return d
