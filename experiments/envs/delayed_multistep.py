"""Two-stage delayed-reward task — multi-step credit assignment probe.

NAME: delayed_multistep | VERSION: 1.0.0

Two corridor stages. Stage 1 (LENGTH1 cells): at t=0 the agent chooses
branch A (action 0) or branch B (action 1); afterwards only `forward`
(action 2) advances; `stay` (action 3) waits. At the end of stage 1 the
agent reaches a junction and must choose a sub-branch (actions 0/1 again).
Stage 2 (LENGTH2 cells): `forward` advances. At the end of stage 2 the
episode terminates with reward +1.0 iff (branch, sub_branch) equals the
hidden 2-bit pattern seeded per episode, else 0.0.

The +1.0 is delivered ~(LENGTH1+LENGTH2) steps after the FIRST decision
that caused it, and it depends on the CONJUNCTION of two temporally
separated choices — a multi-step credit-assignment gap. The pattern is
never in the observation.

Reward semantics:
  +0.02 per forward move (shaping, keeps policies moving; max +0.6)
  +1.00 at stage-2 end iff (branch, sub) == hidden pattern, else +0.00
  No other rewards.

Observation channels:
  pos         : scalar, pos / (LENGTH1+LENGTH2)
  branch      : categorical {0, 1, 2} (2 = none chosen yet)
  sub         : categorical {0, 1, 2} (2 = none chosen yet)
  stage       : categorical {0, 1, 2} (0 = stage 1, 1 = junction, 2 = stage 2)
  steps_left  : scalar, fraction of MAX_STEPS remaining

info carries the hidden pattern for analysis only — agents must not use
info.

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class DelayedMultistep(Environment):
    NAME = "delayed_multistep"
    VERSION = "1.0.0"

    LENGTH1 = 15
    LENGTH2 = 15
    MAX_STEPS = 45
    FORWARD_SHAPING = 0.02
    FINAL_REWARD = 1.0

    def action_space(self):
        return {"type": "discrete", "n": 4,
                "labels": ["branch_a", "branch_b", "forward", "stay"]}

    def observation_space(self):
        return {
            "pos": {"type": "scalar", "low": 0.0, "high": 1.0,
                    "desc": "corridor position / (LENGTH1+LENGTH2)"},
            "branch": {"type": "categorical", "values": [0, 1, 2],
                       "desc": "branch chosen at t=0; 2 = none yet"},
            "sub": {"type": "categorical", "values": [0, 1, 2],
                    "desc": "sub-branch chosen at the junction; 2 = none yet"},
            "stage": {"type": "categorical", "values": [0, 1, 2],
                      "desc": "0 = stage 1, 1 = junction, 2 = stage 2"},
            "steps_left": {"type": "scalar", "low": 0.0, "high": 1.0,
                           "desc": "fraction of MAX_STEPS remaining"},
        }

    @property
    def _total_length(self):
        return self.LENGTH1 + self.LENGTH2

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._pos = 0
        self._branch = None
        self._sub = None
        self._stage = 0
        self._pattern = (self._rng.randrange(2), self._rng.randrange(2))
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        reward = 0.0
        if self._stage == 0:
            # Stage 1: choose branch at t=0, then forward advances.
            if self._pos == 0:
                if action in (0, 1):
                    self._branch = action
                    self._pos = 1
                # forward/stay at t=0: no-op
            else:
                if action == 2 and self._pos < self.LENGTH1:
                    self._pos += 1
                    reward += self.FORWARD_SHAPING
                if self._pos >= self.LENGTH1:
                    self._stage = 1  # junction
        elif self._stage == 1:
            # Junction: must choose the sub-branch (actions 0/1).
            if self._sub is None and action in (0, 1):
                self._sub = action
                self._stage = 2
                self._pos = self.LENGTH1 + 1
            # forward/stay at the junction: no-op
        else:
            # Stage 2: forward advances to the terminal cell.
            if action == 2 and self._pos < self._total_length:
                self._pos += 1
                reward += self.FORWARD_SHAPING
        self._steps += 1

        done, info = False, {}
        if self._pos >= self._total_length:
            done = True
            if (self._branch, self._sub) == self._pattern:
                reward += self.FINAL_REWARD
                info["pattern_match"] = True
            else:
                info["pattern_match"] = False
        elif self._steps >= self.MAX_STEPS:
            done, info["truncated"] = True, True
        # Ground truth for ANALYSIS ONLY — contract forbids agents using info.
        info["pattern"] = list(self._pattern)
        info["stage"] = self._stage
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"pos": self._pos / self._total_length,
               "branch": 2 if self._branch is None else self._branch,
               "sub": 2 if self._sub is None else self._sub,
               "stage": self._stage,
               "steps_left": max(0.0, (self.MAX_STEPS - self._steps)
                                 / self.MAX_STEPS)}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "pos": self._pos, "branch": self._branch, "sub": self._sub,
            "stage": self._stage, "pattern": list(self._pattern),
            "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
