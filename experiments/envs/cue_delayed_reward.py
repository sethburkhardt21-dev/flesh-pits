"""Cue-conditioned delayed-reward task — the K9/K10 intersection probe.

NAME: cue_delayed_reward | VERSION: 1.0.0

Canonical delayed_reward mechanics (v1.0.0) PLUS a per-episode cue in the
observation, constant within the episode:

  cue: categorical {0, 1}, drawn from the episode rng at reset; VISIBLE in
       the observation from t=0 (it is INPUT the agent may use).

The correct branch for the episode is a fixed documented function of the
cue: correct = cue (cue 0 -> branch_a/action 0; cue 1 -> branch_b/action 1).
The mapping is fixed, but the agents under test are stimulus-blind
(neutral specialists, bandit isolation): only input-conditioned gain
machinery (per-cue gain vectors) can encode a cue-conditioned policy, so
the cue must still be read via the input on every episode.

Everything else is identical to canonical delayed_reward v1.0.0:
  corridor of LENGTH cells; at t=0 choose branch A (0) or branch B (1);
  afterwards only `forward` (2) advances; `stay` (3) waits. At the corridor
  end the episode terminates with +1.0 iff the chosen branch matches the
  cue-conditioned correct branch, else 0.0.

Reward semantics:
  +0.02 per forward move (shaping, keeps policies moving; max +0.2)
  +1.00 at corridor end if branch == correct, else +0.00
  The +1.0 is delivered LENGTH steps after the decision that caused it —
  the credit-assignment gap (K9's structural bound), now conditioned on
  a cue (K10's input bound). This env is their intersection.

Observation channels:
  pos         : scalar, pos / LENGTH
  branch      : categorical {0, 1, 2} (2 = none chosen yet)
  steps_left  : scalar, fraction of MAX_STEPS remaining
  cue         : categorical {0, 1}, constant within the episode

info carries the cue/correct branch for analysis only — agents must not
use info.

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class CueDelayedReward(Environment):
    NAME = "cue_delayed_reward"
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
            "cue": {"type": "categorical", "values": [0, 1],
                    "desc": "per-episode cue, constant within the episode; "
                            "correct branch == cue (documented mapping)"},
        }

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._pos = 0
        self._branch = None
        self._cue = self._rng.randrange(2)
        self._correct = self._cue  # documented fixed mapping
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
        info["cue"] = self._cue
        info["correct_branch"] = self._correct
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"pos": self._pos / self.LENGTH,
               "branch": 2 if self._branch is None else self._branch,
               "steps_left": max(0.0, (self.MAX_STEPS - self._steps) /
                                  self.MAX_STEPS),
               "cue": self._cue}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "pos": self._pos, "branch": self._branch, "cue": self._cue,
            "correct": self._correct, "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
