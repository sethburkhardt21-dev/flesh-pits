"""Injected-mismatch self/world environment (additive; self_world untouched).

NAME: self_world_mismatch | VERSION: 1.0.0

Subclass of self_world: hand is SELF-caused, ball WORLD-caused, as before.
ADDITIONAL MECHANISM: on each tick, the env's private RNG decides (unknown
to the agent) whether this tick's action effect is injected as a mismatch
(p = 0.15). Modes (uniform among the three):

  override : hand is moved by a fresh exogenous coin; the action is ignored.
  swap     : hand is moved by an exogenous coin AND ball is moved by the
             agent's own action (+/- STEP) — self- and world-causation are
             simultaneously present on the two channels.
  delay    : hand is HELD (no move) this tick; the action is buffered. Next
             tick (unless another mismatch intervenes), hand moves by the
             STALE buffered action instead of the current one. A buffered
             action overwritten by a new delay is logged (dropped_actions).

Ground-truth cause labels live in info for SCORING ONLY
(contract v1.0.0 section 5):
  info["cause"] = {"hand": "self"|"world", "ball": "self"|"world"}
  info["mismatch"] = {"active": bool, "mode": None|"override"|"swap"|"delay",
                      "delay_pending": bool}

Deterministic given (seed, action sequence). Stdlib only.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_world import SelfWorld  # noqa: E402
from env_interface import canonical_json  # noqa: E402


class SelfWorldMismatch(SelfWorld):
    NAME = "self_world_mismatch"
    VERSION = "1.0.0"
    MISMATCH_P = 0.15

    def reset(self, seed: int) -> dict:
        obs = super().reset(seed)
        self._mm_pending = None       # buffered action for delay mode
        self._mm_delay_hold = False   # True if this tick's move is a delay apply
        self._dropped = 0
        self._last_mismatch = {"active": False, "mode": None,
                               "delay_pending": False}
        return obs

    def _coin(self) -> float:
        return -self.STEP if self._rng.random() < 0.5 else self.STEP

    def step(self, action: int):
        self.validate_action(action)
        d_hand = 0.0
        d_ball = 0.0
        cause_hand = "self"
        cause_ball = "world"
        mode = None
        mismatch = False

        # Delay-apply: a buffered stale action drives hand this tick.
        if self._mm_pending is not None:
            pending = self._mm_pending
            self._mm_pending = None
            d_hand = -self.STEP if pending == 0 else self.STEP
            cause_hand = "world"   # moved by a stale action, not this tick's
            mismatch = True
            mode = "delay"
        else:
            if self._rng.random() < self.MISMATCH_P:
                mismatch = True
                u = self._rng.random()
                if u < 1.0 / 3.0:
                    mode = "override"
                    d_hand = self._coin()
                    cause_hand = "world"
                elif u < 2.0 / 3.0:
                    mode = "swap"
                    d_hand = self._coin()
                    d_ball = -self.STEP if action == 0 else self.STEP
                    cause_hand = "world"
                    cause_ball = "self"
                else:
                    mode = "delay"
                    # Hold hand this tick; buffer the action for next tick.
                    d_hand = 0.0
                    self._mm_pending = action
                    cause_hand = "world"
            else:
                d_hand = -self.STEP if action == 0 else self.STEP

        # Ball moves on its usual exogenous coin except in swap mode
        # (there d_ball already carries the agent's own action effect).
        if mode != "swap":
            d_ball = self._coin()
        self._hand = self._reflect(self._hand + d_hand)
        self._ball = self._reflect(self._ball + d_ball)
        self._steps += 1
        done = self._steps >= self.MAX_STEPS
        # coincides_hand (SCORING ONLY): on a mismatch tick where the hand
        # moved, did the exogenous effect coincide with the commanded
        # direction? A coinciding exogenous move produces residual 0 and is
        # informationally invisible to ANY efference-copy comparator.
        action_sign = 1.0 if action == 1 else -1.0
        coincides_hand = None
        if mismatch and abs(d_hand) > self.STEP / 2.0:
            coincides_hand = (d_hand > 0.0) == (action_sign > 0.0)
        self._last_mismatch = {
            "active": mismatch,
            "mode": mode,
            "delay_pending": self._mm_pending is not None,
            "coincides_hand": coincides_hand,
        }
        info = {
            "cause": {"hand": cause_hand, "ball": cause_ball},
            "mismatch": dict(self._last_mismatch),
            "dropped_delayed_actions": self._dropped,
        }
        if done:
            info["truncated"] = True
        return self._obs(), 0.0, done, info

    # -- introspection ------------------------------------------------------
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
                "pending": self._mm_pending,
                "dropped": self._dropped,
            }).encode("utf-8")).hexdigest()
