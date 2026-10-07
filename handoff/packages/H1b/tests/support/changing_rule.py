"""Changing-rule task — contingency reversal probe.

NAME: changing_rule | VERSION: 1.0.0

A 2-cue, 2-action bandit. Each step presents a cue; reward is +1 if the
action matches the current rule for that cue, else 0 (with 10% noise flips).
The rule (cue -> correct action mapping) is fixed within a phase and flips
to a freshly seeded mapping every PHASE_LEN episodes.

State (documented):
  episode_idx: counts resets on this env object (0-based)
  phase: episode_idx // PHASE_LEN
  rule: [correct_action_for_cue_0, correct_action_for_cue_1], a fixed
        deterministic function of phase (identical schedule every run;
        cue order and noise still vary per reset seed)
  last_reward, last_action: for the observation
  steps: steps taken this episode

Reward semantics: +1.0 correct (90%), 0.0 incorrect (90%); 10% of trials the
reward is flipped (noise). No shaping.

Observation channels:
  cue          : categorical {0, 1}
  last_reward  : scalar {0.0, 1.0}, 0.0 on the first step
  last_action  : categorical {0, 1, 2} (2 = none yet)

The rule itself is NEVER in the observation. info carries the ground-truth
rule/phase/flip flag for analysis only — agents must not use info.

This is the adaptation probe: after a flip, a learning agent must detect
the contingency change from reward prediction errors and re-learn. A static
policy cannot recover.

Deterministic given (seed, episode_idx, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class ChangingRule(Environment):
    NAME = "changing_rule"
    VERSION = "1.0.0"

    N_CUES = 2
    STEPS = 40
    PHASE_LEN = 5
    NOISE = 0.10

    def action_space(self):
        return {"type": "discrete", "n": 2, "labels": ["a0", "a1"]}

    def observation_space(self):
        return {
            "cue": {"type": "categorical", "values": [0, 1],
                    "desc": "current cue"},
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
        # NOTE: the rule schedule is a fixed deterministic function of phase
        # (identical for every run). Per-episode stochasticity (cue order,
        # noise) still varies with the reset seed. This keeps the contingency
        # schedule comparable across agents and runs.
        rule_rng = new_rng((phase * 7919 + 0x5EED) % (2 ** 31))
        self._rule = [rule_rng.randrange(2) for _ in range(self.N_CUES)]
        self._flipped = (self._last_phase is not None and phase != self._last_phase)
        self._last_phase = phase
        self._phase = phase
        self._last_reward = 0.0
        self._last_action = 2
        self._cue = self._rng.randrange(self.N_CUES)
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        correct = self._rule[self._cue]
        reward = 1.0 if action == correct else 0.0
        if self._rng.random() < self.NOISE:
            reward = 1.0 - reward
        self._last_reward = reward
        self._last_action = action
        self._steps += 1
        self._cue = self._rng.randrange(self.N_CUES)

        done = self._steps >= self.STEPS
        info = {"truncated": True} if done else {}
        # Ground truth for ANALYSIS ONLY — contract forbids agents using info.
        info.update({"phase": self._phase, "rule": list(self._rule),
                     "flipped_this_episode": self._flipped,
                     "episode_idx": self._episode_idx})
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"cue": self._cue,
               "last_reward": self._last_reward,
               "last_action": self._last_action}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "episode_idx": self._episode_idx, "phase": self._phase,
            "rule": self._rule, "cue": self._cue,
            "last_reward": self._last_reward, "last_action": self._last_action,
            "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
