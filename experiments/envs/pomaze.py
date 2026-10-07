"""Partial-observability maze — memory-dependent navigation.

NAME: pomaze | VERSION: 1.0.0

A perfect maze (randomized DFS, seeded) on an odd-sized grid. The agent
starts at (1,1); the goal is at (SIZE-2, SIZE-2). The maze is always fully
connected.

State (documented):
  walls: perfect-maze wall set (seeded)
  pos: agent (x, y)
  steps: steps taken this episode

Reward semantics:
  -0.01 per step, -0.02 on wall bump (stay in place), +1.00 at goal (ends).

Observation channels — deliberately PARTIAL:
  wall_n, wall_s, wall_e, wall_w : binary, walls adjacent to current cell
  beacon   : coarse scent gradient toward goal, quantized to {0, 1/3, 2/3, 1}
             (1 = at/near goal). Locally ambiguous; loop-avoidance requires
             memory of where the agent has been.
  steps_left : fraction of MAX_STEPS remaining

There is NO global position and NO goal direction. A purely reactive policy
can follow the beacon gradient but cannot avoid revisiting cells; systematic
search requires memory. This is the memory-dependence probe (§35).

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class POMaze(Environment):
    NAME = "pomaze"
    VERSION = "1.0.0"

    SIZE = 11  # must be odd
    MAX_STEPS = 200
    STEP_PENALTY = -0.01
    BUMP_PENALTY = -0.02
    GOAL_REWARD = 1.0
    DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W

    def action_space(self):
        return {"type": "discrete", "n": 4,
                "labels": ["north", "south", "east", "west"]}

    def observation_space(self):
        return {
            "wall_n": {"type": "scalar", "low": 0.0, "high": 1.0,
                       "desc": "1 if wall north of agent"},
            "wall_s": {"type": "scalar", "low": 0.0, "high": 1.0,
                       "desc": "1 if wall south of agent"},
            "wall_e": {"type": "scalar", "low": 0.0, "high": 1.0,
                       "desc": "1 if wall east of agent"},
            "wall_w": {"type": "scalar", "low": 0.0, "high": 1.0,
                       "desc": "1 if wall west of agent"},
            "beacon": {"type": "scalar", "low": 0.0, "high": 1.0,
                       "desc": "coarse goal proximity in {0,1/3,2/3,1}"},
            "steps_left": {"type": "scalar", "low": 0.0, "high": 1.0,
                           "desc": "fraction of MAX_STEPS remaining"},
        }

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._walls = self._gen_maze()
        self._start = (1, 1)
        self._goal = (self.SIZE - 2, self.SIZE - 2)
        self._pos = self._start
        self._max_dist = 2 * (self.SIZE - 1)
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        dx, dy = self.DELTAS[action]
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        reward = self.STEP_PENALTY
        if (nx, ny) in self._walls:
            reward += self.BUMP_PENALTY  # stay in place
        else:
            self._pos = (nx, ny)
        self._steps += 1

        done, info = False, {}
        if self._pos == self._goal:
            reward += self.GOAL_REWARD
            done, info["goal_reached"] = True, True
        elif self._steps >= self.MAX_STEPS:
            done, info["truncated"] = True, True
        return self._obs(), reward, done, info

    # -- internals --------------------------------------------------------
    def _gen_maze(self):
        n = self.SIZE
        walls = {(x, y) for y in range(n) for x in range(n)}
        stack = [(1, 1)]
        walls.discard((1, 1))
        while stack:
            x, y = stack[-1]
            options = []
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                nx, ny = x + dx, y + dy
                if 1 <= nx < n - 1 and 1 <= ny < n - 1 and (nx, ny) in walls:
                    options.append((nx, ny, dx // 2, dy // 2))
            if options:
                nx, ny, wx, wy = self._rng.choice(options)
                walls.discard((nx, ny))
                walls.discard((x + wx, y + wy))
                stack.append((nx, ny))
            else:
                stack.pop()
        return walls

    def _is_wall(self, x, y):
        return (x, y) in self._walls

    def _obs(self):
        x, y = self._pos
        dist = abs(x - self._goal[0]) + abs(y - self._goal[1])
        level = round(3 * (1.0 - dist / self._max_dist)) / 3.0
        obs = {
            "wall_n": 1.0 if self._is_wall(x, y - 1) else 0.0,
            "wall_s": 1.0 if self._is_wall(x, y + 1) else 0.0,
            "wall_e": 1.0 if self._is_wall(x + 1, y) else 0.0,
            "wall_w": 1.0 if self._is_wall(x - 1, y) else 0.0,
            "beacon": level,
            "steps_left": max(0.0, (self.MAX_STEPS - self._steps) / self.MAX_STEPS),
        }
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "walls": sorted(self._walls),
            "pos": self._pos, "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
