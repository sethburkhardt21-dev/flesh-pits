"""Odometry observation wrapper — EXP-FP-0011-ALIASING (additive).

Appends a coarse position-disambiguating feature to any env's observation,
computed by driver-side dead reckoning from the OBSERVATION STREAM and the
chosen actions only.

Legitimacy contract (frozen in preregistration_ALIASING.json):
  * NO goal position, NO maze walls, NO absolute position, NO env internals.
  * pos_rel starts at (0,0) at reset (relative only — the documented start
    cell is never used).
  * On each step with chosen action a: if the pre-step obs carries the
    wall_n/wall_s/wall_e/wall_w fields and the wall sensor in direction a is
    0 (no wall), pos_rel += DELTAS[a]; otherwise unchanged. Envs without
    wall fields (e.g. changing_rule) leave pos_rel at (0,0) -> the channels
    are constant 0.0 (inert augmentation).
  * Appended channels: odom_qx, odom_qy = clamp((pos_rel // CELL + 2) / 4,
    0, 1) with CELL = 3 — a coarse 4x4 relative-position hash.
  * drive="true": odometry follows the true chosen actions.
    drive="shuffle": odometry follows an independent seeded random action
    stream (new_rng(derive_seed(ep_seed, run_index, "shuffle"))),
    decorrelated from true position — the §40 kill/lesion arm: same channel
    format/range, no disambiguation information.

Stdlib only. Deterministic given (seed, run_index, action sequence).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import derive_seed, new_rng  # noqa: E402


class OdomWrapper:
    """Driver-side additive obs wrapper. B's core is untouched."""

    DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W (matches POMaze)
    WALL_FIELDS = ["wall_n", "wall_s", "wall_e", "wall_w"]
    CELL = 3

    def __init__(self, env, drive="true"):
        if drive not in ("true", "shuffle"):
            raise ValueError(f"unknown drive {drive!r}")
        self.env = env
        self.drive = drive
        self.NAME = env.NAME + "+odom"
        self.VERSION = env.VERSION
        self._x = 0
        self._y = 0
        self._last_obs = None
        self._shuffle_rng = None

    # -- spaces ---------------------------------------------------------
    def action_space(self):
        return self.env.action_space()

    def observation_space(self):
        space = dict(self.env.observation_space())
        space["odom_qx"] = {
            "type": "scalar", "low": 0.0, "high": 1.0,
            "desc": "coarse relative x cell from odometry (legitimate)"}
        space["odom_qy"] = {
            "type": "scalar", "low": 0.0, "high": 1.0,
            "desc": "coarse relative y cell from odometry (legitimate)"}
        return space

    # -- lifecycle ------------------------------------------------------
    def reset(self, seed: int, run_index: int = 0) -> dict:
        self._x, self._y = 0, 0
        self._ep_seed = seed
        if self.drive == "shuffle":
            self._shuffle_rng = new_rng(
                derive_seed(seed, run_index, "shuffle"))
        obs = self.env.reset(seed)
        self._last_obs = obs
        return self._augment(obs)

    def step(self, action: int):
        self.env.validate_action(action)
        pre = self._last_obs
        if all(f in pre for f in self.WALL_FIELDS):
            if self.drive == "true":
                move_a = action
            else:
                move_a = self._shuffle_rng.randrange(4)
            if pre[self.WALL_FIELDS[move_a]] < 0.5:
                dx, dy = self.DELTAS[move_a]
                self._x += dx
                self._y += dy
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        return self._augment(obs), reward, done, info

    # -- augmentation -----------------------------------------------------
    def _augment(self, obs: dict) -> dict:
        cx = self._x // self.CELL
        cy = self._y // self.CELL
        aug = dict(obs)
        aug["odom_qx"] = min(1.0, max(0.0, (cx + 2) / 4.0))
        aug["odom_qy"] = min(1.0, max(0.0, (cy + 2) / 4.0))
        return self.env.validate_obs(aug, self.observation_space())
