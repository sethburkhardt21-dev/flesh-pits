"""Grid world — navigation with obstacles, hazards, and a goal.

NAME: grid_world | VERSION: 1.0.0

State (documented):
  layout: walls set, goal cell, hazard cells (seeded; connectivity guaranteed
          by BFS check: goal reachable from start with path length >= 8)
  pos: agent (x, y)
  steps: steps taken this episode

Reward semantics:
  -0.02 per step (time pressure)
  -0.05 on wall bump / out-of-bounds (agent stays in place)
  +1.00 on reaching the goal (episode ends)
  -0.50 on entering a hazard cell (episode ends)

Observation channels (all in [0,1] or [-1,1] as documented):
  pos_x, pos_y   agent position normalized by (SIZE-1)
  goal_dx, goal_dy  goal minus agent, normalized by (SIZE-1), in [-1,1]
  view           3x3 local view, row-major, cell code/3
                 (0=empty, 1=wall incl. out-of-bounds, 2=goal, 3=hazard;
                  center cell forced to 0 = own position)
  steps_left     fraction of MAX_STEPS remaining

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class GridWorld(Environment):
    NAME = "grid_world"
    VERSION = "1.0.0"

    SIZE = 8
    MAX_STEPS = 100
    WALL_PROB = 0.12
    N_HAZARDS = 2
    MIN_GOAL_DIST = 8
    STEP_PENALTY = -0.02
    BUMP_PENALTY = -0.05
    GOAL_REWARD = 1.0
    HAZARD_REWARD = -0.5

    EMPTY, WALL, GOAL, HAZARD = 0, 1, 2, 3
    DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]  # N S E W stay

    # -- spaces -----------------------------------------------------------
    def action_space(self):
        return {"type": "discrete", "n": 5,
                "labels": ["north", "south", "east", "west", "stay"]}

    def observation_space(self):
        return {
            "pos_x": {"type": "scalar", "low": 0.0, "high": 1.0,
                      "desc": "agent x / (SIZE-1)"},
            "pos_y": {"type": "scalar", "low": 0.0, "high": 1.0,
                      "desc": "agent y / (SIZE-1)"},
            "goal_dx": {"type": "scalar", "low": -1.0, "high": 1.0,
                        "desc": "(goal_x - agent_x) / (SIZE-1)"},
            "goal_dy": {"type": "scalar", "low": -1.0, "high": 1.0,
                        "desc": "(goal_y - agent_y) / (SIZE-1)"},
            "view": {"type": "vector", "shape": 9, "low": 0.0, "high": 1.0,
                     "desc": "3x3 local view row-major, cell code / 3, center = 0"},
            "steps_left": {"type": "scalar", "low": 0.0, "high": 1.0,
                           "desc": "fraction of MAX_STEPS remaining"},
        }

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._layout = self._gen_layout()
        self._pos = (0, 0)
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        dx, dy = self.DELTAS[action]
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        reward = self.STEP_PENALTY
        if (dx, dy) == (0, 0):
            pass  # stay: no bump penalty
        elif not (0 <= nx < self.SIZE and 0 <= ny < self.SIZE) \
                or (nx, ny) in self._layout["walls"]:
            reward += self.BUMP_PENALTY
        else:
            self._pos = (nx, ny)
        self._steps += 1

        done, info = False, {}
        if self._pos == self._layout["goal"]:
            reward += self.GOAL_REWARD
            done, info["goal_reached"] = True, True
        elif self._pos in self._layout["hazards"]:
            reward += self.HAZARD_REWARD
            done, info["hazard_hit"] = True, True
        elif self._steps >= self.MAX_STEPS:
            done, info["truncated"] = True, True
        return self._obs(), reward, done, info

    # -- internals --------------------------------------------------------
    def _gen_layout(self):
        for _ in range(500):
            walls = set()
            for y in range(self.SIZE):
                for x in range(self.SIZE):
                    if (x, y) == (0, 0):
                        continue
                    if self._rng.random() < self.WALL_PROB:
                        walls.add((x, y))
            dist = self._bfs_dist(walls, (0, 0))
            free = [(x, y) for y in range(self.SIZE) for x in range(self.SIZE)
                    if (x, y) not in walls and (x, y) != (0, 0)]
            cands = [c for c in free if dist.get(c, -1) >= self.MIN_GOAL_DIST]
            if not cands:
                continue
            goal = self._rng.choice(cands)
            rest = [c for c in free if c != goal]
            hazards = set(self._rng.sample(rest, min(self.N_HAZARDS, len(rest))))
            return {"walls": walls, "goal": goal, "hazards": hazards}
        raise RuntimeError("grid_world: layout generation failed after 500 attempts")

    def _bfs_dist(self, walls, start):
        dist = {start: 0}
        dq = deque([start])
        while dq:
            x, y = dq.popleft()
            for dx, dy in self.DELTAS[:4]:
                nb = (x + dx, y + dy)
                if 0 <= nb[0] < self.SIZE and 0 <= nb[1] < self.SIZE \
                        and nb not in walls and nb not in dist:
                    dist[nb] = dist[(x, y)] + 1
                    dq.append(nb)
        return dist

    def _cell(self, x, y):
        if not (0 <= x < self.SIZE and 0 <= y < self.SIZE):
            return self.WALL
        if (x, y) in self._layout["walls"]:
            return self.WALL
        if (x, y) == self._layout["goal"]:
            return self.GOAL
        if (x, y) in self._layout["hazards"]:
            return self.HAZARD
        return self.EMPTY

    def _obs(self):
        x, y = self._pos
        gx, gy = self._layout["goal"]
        view = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    view.append(0.0)
                else:
                    view.append(self._cell(x + dx, y + dy) / 3.0)
        obs = {
            "pos_x": x / (self.SIZE - 1),
            "pos_y": y / (self.SIZE - 1),
            "goal_dx": (gx - x) / (self.SIZE - 1),
            "goal_dy": (gy - y) / (self.SIZE - 1),
            "view": tuple(view),
            "steps_left": max(0.0, (self.MAX_STEPS - self._steps) / self.MAX_STEPS),
        }
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "walls": sorted(self._layout["walls"]),
            "goal": self._layout["goal"],
            "hazards": sorted(self._layout["hazards"]),
            "pos": self._pos, "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
