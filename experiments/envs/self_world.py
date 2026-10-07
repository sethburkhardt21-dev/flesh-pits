"""Self/world distinction probe environment.

NAME: self_world | VERSION: 1.0.0

Two scalar channels on [0, 1]:
  hand : SELF-caused.  hand_{t+1} = reflect(hand_t +/- STEP) with the sign
         given by the agent's own action (0=left -> -STEP, 1=right -> +STEP).
         Deterministic given (hand_t, action).
  ball : WORLD-caused. ball_{t+1} = reflect(ball_t +/- STEP) with the sign
         drawn from a fair coin on the env's private RNG stream.
         Exogenous: no function of the agent's action.

Matched statistics (by construction — the distinction must require causal
attribution, not correlation):
  - both channels change with |displacement| in {0, STEP}: STEP normally,
    0 only at the reflection fixed points (STEP/2 and 1-STEP/2), where a
    step reflects onto itself — identical for both channels;
  - direction marginals 50/50 under a balanced action stream (hand) and
    by construction (ball, fair coin).

The ONLY structural difference is controllability: hand increments are a
known deterministic function of the agent's own action; ball increments
come from an unobserved exogenous stream.

Reward semantics: 0.0 on every tick (the probe measures representational
distinction, not task performance; a constant reward keeps reward-driven
machinery symmetric across channels).

info carries ground-truth cause labels for SCORING ONLY:
  {"cause": {"hand": "self", "ball": "world"}, ...}
Agents are contractually forbidden from using info (contract v1.0.0 §5);
the probe script (not the agent) reads it to label transitions.

Deterministic given (seed, action sequence). Stdlib only.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import (  # noqa: E402
    Environment, new_rng, canonical_json,
)


class SelfWorld(Environment):
    NAME = "self_world"
    VERSION = "1.0.0"

    STEP = 0.2
    MAX_STEPS = 60

    def action_space(self):
        return {"type": "discrete", "n": 2, "labels": ["left", "right"]}

    def observation_space(self):
        return {
            "hand": {"type": "scalar", "low": 0.0, "high": 1.0,
                     "desc": "SELF-caused channel: moved deterministically "
                             "by the agent's own action (+/- STEP)"},
            "ball": {"type": "scalar", "low": 0.0, "high": 1.0,
                     "desc": "WORLD-caused channel: moved by an exogenous "
                             "fair-coin event (+/- STEP)"},
        }

    # -- lifecycle ------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._steps = 0
        self._hand = 0.5
        self._ball = 0.5
        self._seed = int(seed)
        return self._obs()

    def _reflect(self, v: float) -> float:
        if v < 0.0:
            return -v
        if v > 1.0:
            return 2.0 - v
        return v

    def _obs(self) -> dict:
        return {"hand": self._hand, "ball": self._ball}

    def step(self, action: int):
        self.validate_action(action)
        # SELF: hand moves deterministically from the agent's own action.
        d_hand = -self.STEP if action == 0 else self.STEP
        # WORLD: ball moves from an exogenous fair coin (unobserved by agent).
        d_ball = -self.STEP if self._rng.random() < 0.5 else self.STEP
        self._hand = self._reflect(self._hand + d_hand)
        self._ball = self._reflect(self._ball + d_ball)
        self._steps += 1
        done = self._steps >= self.MAX_STEPS
        info = {"cause": {"hand": "self", "ball": "world"}}
        if done:
            info["truncated"] = True
        return self._obs(), 0.0, done, info

    # -- introspection --------------------------------------------------
    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json({
                "name": self.NAME,
                "version": self.VERSION,
                "seed": self._seed,
                "steps": self._steps,
                "hand": self._hand,
                "ball": self._ball,
                "rng_state": self._rng.getstate(),
            }).encode("utf-8")).hexdigest()
