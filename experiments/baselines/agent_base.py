"""Shared scaffolding for the §39 baseline agents. Stdlib only.

Contract notes (see ENV_INTERFACE.md):
- reset(seed, action_space) reinitializes PER-EPISODE state (RNG, counters).
  Cross-episode learning persists in the agent OBJECT: the harness reuses one
  agent instance across all episodes of a run, calling reset() each episode.
- act() and update() must NEVER use `info` for decisions (metadata only).
- All randomness comes from the per-episode seeded RNG (self._rng).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Agent, new_rng  # noqa: E402


def flatten_obs(obs: dict) -> tuple:
    """Deterministic flattening of an observation dict to a float tuple.

    Sorted key order; nested vectors extended elementwise. Categorical
    integer codes are used as-is (numeric). Pure function of obs.
    """
    out = []
    for key in sorted(obs.keys()):
        v = obs[key]
        if isinstance(v, (list, tuple)):
            out.extend(float(x) for x in v)
        else:
            out.append(float(v))
    return tuple(out)


def argmax_det(scores) -> int:
    """Argmax with deterministic tie-break (lowest index). No randomness."""
    best, best_i = scores[0], 0
    for i, s in enumerate(scores[1:], 1):
        if s > best:
            best, best_i = s, i
    return best_i


def rng_state_list(rng):
    st = rng.getstate()
    return [st[0], list(st[1]), st[2]]


def rng_set_from_list(rng, lst):
    rng.setstate((lst[0], tuple(lst[1]), lst[2]))


class BaselineAgent(Agent):
    """Common scaffolding: seeded RNG, action count, snapshot/restore."""

    def __init__(self):
        self._rng = new_rng(0)
        self._n_actions = 2
        self._seed = 0

    # -- Agent interface -------------------------------------------------
    def reset(self, seed: int, action_space: dict) -> None:
        self._seed = seed
        self._rng = new_rng(seed)
        self._n_actions = action_space["n"]
        self._on_reset()

    def _on_reset(self):
        """Per-episode state init. Override; do NOT wipe cross-episode learning."""

    def act(self, obs: dict) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        """Default: no learning. Override where the baseline learns."""

    # -- continuity (§29) --------------------------------------------------
    def snapshot(self) -> dict:
        state = {"agent": self.NAME, "seed": self._seed,
                 "n_actions": self._n_actions,
                 "rng": rng_state_list(self._rng)}
        state.update(self._extra_state())
        return state

    def restore(self, state: dict) -> None:
        assert state["agent"] == self.NAME, "snapshot/agent mismatch"
        self._seed = state["seed"]
        self._n_actions = state["n_actions"]
        self._rng = new_rng(0)
        rng_set_from_list(self._rng, state["rng"])
        self._restore_extra(state)

    def _extra_state(self) -> dict:
        return {}

    def _restore_extra(self, state: dict) -> None:
        pass


class TransitionLearner:
    """One-step-delayed transition learning inside the act/update contract.

    The contract's update() receives (obs_t, action_t, reward_t) but NOT
    obs_{t+1}; the next act() receives obs_{t+1}. So this mixin:

    - update(): completes the PENDING transition (t-1 -> current obs) via
      learn_transition(), then either stashes the current triple as the new
      pending or, on done, learns the terminal transition immediately with
      next_obs = zero vector (documented convention: terminal obs ≡ 0).
    - act(): pure decision, no learning.

    Every transition is learned exactly once. Subclasses implement
    learn_transition(obs_vec, action, next_obs_vec, reward, done).
    """

    def _tl_reset(self):
        self._pending = None

    def _tl_update(self, obs_vec, action, reward, done, obs_dim):
        if self._pending is not None:
            po, pa, pr = self._pending
            self.learn_transition(po, pa, tuple(obs_vec), pr, False)
            self._pending = None
        if done:
            self.learn_transition(tuple(obs_vec), action,
                                  (0.0,) * obs_dim, reward, True)
        else:
            self._pending = (tuple(obs_vec), action, reward)

    def _tl_state(self):
        p = self._pending
        return {"pending": None if p is None else [list(p[0]), p[1], p[2]]}

    def _tl_restore(self, state):
        p = state.get("pending")
        self._pending = None if p is None else (tuple(p[0]), p[1], p[2])

    def learn_transition(self, obs_vec, action, next_obs_vec, reward, done):
        raise NotImplementedError
