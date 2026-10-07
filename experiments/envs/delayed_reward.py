"""Delayed-reward task — temporal credit assignment probe.

NAME: delayed_reward | VERSION: 1.0.0

A corridor of LENGTH cells. At t=0 the agent chooses branch A (action 0) or
branch B (action 1). Afterwards only `forward` (action 2) advances; `stay`
(action 3) waits. At the corridor end the episode terminates with reward
+1.0 iff the chosen branch matches the hidden correct branch (seeded per
episode), else 0.0.

State (documented):
  pos: corridor position 0..LENGTH (LENGTH = terminal)
  branch: None | 0 | 1 (choice made at t=0)
  correct: hidden correct branch for this episode (seeded)
  steps: steps taken

Reward semantics:
  +0.02 per forward move (shaping, keeps policies moving; max +0.2)
  +1.00 at corridor end if branch == correct, else +0.00
  No other rewards. The +1.0 is delivered LENGTH steps after the decision
  that caused it — the credit-assignment gap.

Observation channels:
  pos         : scalar, pos / LENGTH
  branch      : categorical {0, 1, 2} (2 = none chosen yet)
  steps_left  : fraction of MAX_STEPS remaining

info carries the correct branch for analysis only — agents must not use info.

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class DelayedReward(Environment):
    NAME = "delayed_reward"
    VERSION = "1.0.0"

    LENGTH = 10
    MAX_STEPS = 15
    FORWARD_SHAPING = 0.02
    FINAL_REWARD = 1.0

    def action_space(self):
        return {"type": "discrete", "n": 4,
                "labels": ["branch_a", "branch_b", "forward", "stay"]}

    def observation_space(self):
        return {
            "pos": {"type": "scalar", "low": 0.0, "high": 1.0,
                    "desc": "corridor position / LENGTH"},
            "branch": {"type": "categorical", "values": [0, 1, 2],
                       "desc": "branch chosen at t=0; 2 = none yet"},
            "steps_left": {"type": "scalar", "low": 0.0, "high": 1.0,
                           "desc": "fraction of MAX_STEPS remaining"},
        }

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._pos = 0
        self._branch = None
        self._correct = self._rng.randrange(2)
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        reward = 0.0
        if self._pos == 0:
            if action in (0, 1):
                self._branch = action
                self._pos = 1
            # forward/stay at t=0: no-op
        else:
            if action == 2 and self._pos < self.LENGTH:
                self._pos += 1
                reward += self.FORWARD_SHAPING
        self._steps += 1

        done, info = False, {}
        if self._pos >= self.LENGTH:
            done = True
            if self._branch == self._correct:
                reward += self.FINAL_REWARD
                info["branch_correct"] = True
            else:
                info["branch_correct"] = False
        elif self._steps >= self.MAX_STEPS:
            done, info["truncated"] = True, True
        # Ground truth for ANALYSIS ONLY — contract forbids agents using info.
        info["correct_branch"] = self._correct
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"pos": self._pos / self.LENGTH,
               "branch": 2 if self._branch is None else self._branch,
               "steps_left": max(0.0, (self.MAX_STEPS - self._steps) / self.MAX_STEPS)}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "pos": self._pos, "branch": self._branch, "correct": self._correct,
            "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
