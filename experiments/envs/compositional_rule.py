"""Compositional-rule task — nonlinear contingency probe.

NAME: compositional_rule | VERSION: 1.0.0

A 2-cue, 2-action bandit. Each step presents two independent cues
(cue_a, cue_b in {0, 1}). The correct action is a COMPOSITION::

    correct = map_a[cue_a] XOR map_b[cue_b]

where map_a, map_b are per-phase binary tables drawn from {(0,1), (1,0)}
(never constant — the contingency is always a genuine 3-way interaction
of cue_a x cue_b x action, never reducible to a single cue). The maps are
a fixed deterministic function of phase and flip to freshly drawn maps
every PHASE_LEN episodes.

Single-cue marginals are uninformative by construction: each cue alone
predicts the correct action at exactly 50%. Only a context-indexed
representation (one expert per (cue_a, cue_b) context) can capture the
contingency — a global linear head cannot represent the XOR.

Reward semantics: +1.0 correct, 0.0 incorrect, with 10% noise flips.
No shaping.

Observation channels:
  cue_a        : categorical {0, 1}
  cue_b        : categorical {0, 1}
  last_reward  : scalar {0.0, 1.0}, 0.0 on the first step
  last_action  : categorical {0, 1, 2} (2 = none yet)

The maps are NEVER in the observation. info carries the maps/phase for
analysis only — agents must not use info.

This is the compositional probe: after a flip, a learning agent must
detect the contingency change from reward prediction errors and re-learn
a 3-way interaction. A linear predictor cannot represent it at all.

Deterministic given (seed, episode_idx, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class CompositionalRule(Environment):
    NAME = "compositional_rule"
    VERSION = "1.0.0"

    N_CUES = 2
    STEPS = 40
    PHASE_LEN = 5
    NOISE = 0.10
    # Non-constant 2-bit maps only: the contingency is always compositional.
    MAPS = ((0, 1), (1, 0))

    def action_space(self):
        return {"type": "discrete", "n": 2, "labels": ["a0", "a1"]}

    def observation_space(self):
        return {
            "cue_a": {"type": "categorical", "values": [0, 1],
                      "desc": "first cue"},
            "cue_b": {"type": "categorical", "values": [0, 1],
                      "desc": "second cue"},
            "last_reward": {"type": "scalar", "low": 0.0, "high": 1.0,
                            "desc": "reward of previous step, 0 at t=0"},
            "last_action": {"type": "categorical", "values": [0, 1, 2],
                            "desc": "previous action, 2 = none yet"},
        }

    # -- lifecycle --------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._episode_idx = -1
        self._last_phase = None

    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._episode_idx += 1
        phase = self._episode_idx // self.PHASE_LEN
        # NOTE: the map schedule is a fixed deterministic function of phase
        # (identical for every run), mirroring changing_rule's rule schedule.
        map_rng = new_rng((phase * 7919 + 0x5EED) % (2 ** 31))
        self._map_a = list(self.MAPS[map_rng.randrange(2)])
        self._map_b = list(self.MAPS[map_rng.randrange(2)])
        self._flipped = (self._last_phase is not None
                         and phase != self._last_phase)
        self._last_phase = phase
        self._phase = phase
        self._last_reward = 0.0
        self._last_action = 2
        self._cue_a = self._rng.randrange(self.N_CUES)
        self._cue_b = self._rng.randrange(self.N_CUES)
        return self._obs()

    def _correct(self):
        return self._map_a[self._cue_a] ^ self._map_b[self._cue_b]

    def step(self, action: int):
        action = self.validate_action(action)
        reward = 1.0 if action == self._correct() else 0.0
        if self._rng.random() < self.NOISE:
            reward = 1.0 - reward
        self._last_reward = reward
        self._last_action = action
        self._steps += 1
        self._cue_a = self._rng.randrange(self.N_CUES)
        self._cue_b = self._rng.randrange(self.N_CUES)

        done = self._steps >= self.STEPS
        info = {"truncated": True} if done else {}
        # Ground truth for ANALYSIS ONLY — contract forbids agents using info.
        info.update({"phase": self._phase, "map_a": list(self._map_a),
                     "map_b": list(self._map_b),
                     "flipped_this_episode": self._flipped,
                     "episode_idx": self._episode_idx})
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"cue_a": self._cue_a,
               "cue_b": self._cue_b,
               "last_reward": self._last_reward,
               "last_action": self._last_action}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "episode_idx": self._episode_idx, "phase": self._phase,
            "map_a": self._map_a, "map_b": self._map_b,
            "cue_a": self._cue_a, "cue_b": self._cue_b,
            "last_reward": self._last_reward, "last_action": self._last_action,
            "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
