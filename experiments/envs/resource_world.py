"""Resource-maintenance world — homeostasis probe.

NAME: resource_world | VERSION: 1.0.0

A SIZE x SIZE grid (border walls only). The agent has two internal resource
levels, satiety and hydration, each decaying per step. Food cells restore
satiety to 1.0; water cells restore hydration to 1.0. Consumed cells respawn
elsewhere after RESPAWN_DELAY steps. If either level reaches 0 the agent
dies (episode ends, -1.0).

State (documented):
  pos: agent (x, y)
  satiety, hydration: floats in [0, 1]
  resources: dict pos -> "food" | "water"
  respawn_queue: list of (due_step, kind)
  steps: steps taken this episode

Reward semantics:
  +0.02 * min(satiety, hydration) per step alive (homeostatic reward:
   staying balanced pays; the worse-off resource dominates)
  -1.00 on death (either level hits 0)
  -0.01 on wall bump (stay in place)
  No reward for eating per se — only for the resulting balance.

Observation channels:
  view      : 3x3 local view row-major, cell code / 3
              (0=empty, 1=wall incl. border, 2=food, 3=water; center = 0)
  satiety   : scalar [0,1]
  hydration : scalar [0,1]
  steps_left: fraction of MAX_STEPS remaining

The agent must learn to seek resources BEFORE depletion — a minimal
allostatic regulation loop over grounded channels (§33 / matrix row 17).

Deterministic given (seed, action sequence). Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import Environment, new_rng, canonical_json  # noqa: E402
import hashlib  # noqa: E402


class ResourceWorld(Environment):
    NAME = "resource_world"
    VERSION = "1.0.0"

    SIZE = 7
    MAX_STEPS = 200
    DECAY = 0.04
    N_FOOD = 3
    N_WATER = 3
    RESPAWN_DELAY = 20
    BUMP_PENALTY = -0.01
    DEATH_PENALTY = -1.0
    ALIVE_RATE = 0.02

    EMPTY, WALL, FOOD, WATER = 0, 1, 2, 3
    DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W

    def action_space(self):
        return {"type": "discrete", "n": 4,
                "labels": ["north", "south", "east", "west"]}

    def observation_space(self):
        return {
            "view": {"type": "vector", "shape": 9, "low": 0.0, "high": 1.0,
                     "desc": "3x3 local view row-major, cell code / 3, center = 0"},
            "satiety": {"type": "scalar", "low": 0.0, "high": 1.0,
                        "desc": "food resource level"},
            "hydration": {"type": "scalar", "low": 0.0, "high": 1.0,
                          "desc": "water resource level"},
            "steps_left": {"type": "scalar", "low": 0.0, "high": 1.0,
                           "desc": "fraction of MAX_STEPS remaining"},
        }

    # -- lifecycle --------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        c = self.SIZE // 2
        self._pos = (c, c)
        self._satiety = 1.0
        self._hydration = 1.0
        self._resources = {}
        self._respawn_queue = []
        free = [(x, y) for y in range(1, self.SIZE - 1)
                for x in range(1, self.SIZE - 1) if (x, y) != self._pos]
        chosen = self._rng.sample(free, self.N_FOOD + self.N_WATER)
        for p in chosen[:self.N_FOOD]:
            self._resources[p] = "food"
        for p in chosen[self.N_FOOD:]:
            self._resources[p] = "water"
        return self._obs()

    def step(self, action: int):
        action = self.validate_action(action)
        dx, dy = self.DELTAS[action]
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        reward = 0.0
        if not (1 <= nx < self.SIZE - 1 and 1 <= ny < self.SIZE - 1):
            reward += self.BUMP_PENALTY  # border wall: stay
        else:
            self._pos = (nx, ny)

        self._satiety = max(0.0, self._satiety - self.DECAY)
        self._hydration = max(0.0, self._hydration - self.DECAY)

        kind = self._resources.pop(self._pos, None)
        if kind == "food":
            self._satiety = 1.0
            self._respawn_queue.append((self._steps + self.RESPAWN_DELAY, "food"))
        elif kind == "water":
            self._hydration = 1.0
            self._respawn_queue.append((self._steps + self.RESPAWN_DELAY, "water"))
        self._do_respawns()

        self._steps += 1
        done, info = False, {}
        if self._satiety <= 0.0 or self._hydration <= 0.0:
            reward += self.DEATH_PENALTY
            done = True
            info["death"] = "satiety" if self._satiety <= 0.0 else "hydration"
        elif self._steps >= self.MAX_STEPS:
            done, info["truncated"] = True, True
        else:
            reward += self.ALIVE_RATE * min(self._satiety, self._hydration)
        return self._obs(), reward, done, info

    # -- internals --------------------------------------------------------
    def _do_respawns(self):
        due = [q for q in self._respawn_queue if q[0] <= self._steps]
        self._respawn_queue = [q for q in self._respawn_queue if q[0] > self._steps]
        for _, kind in due:
            free = [(x, y) for y in range(1, self.SIZE - 1)
                    for x in range(1, self.SIZE - 1)
                    if (x, y) != self._pos and (x, y) not in self._resources]
            if free:
                self._resources[self._rng.choice(free)] = kind

    def _cell(self, x, y):
        if not (1 <= x < self.SIZE - 1 and 1 <= y < self.SIZE - 1):
            return self.WALL
        kind = self._resources.get((x, y))
        if kind == "food":
            return self.FOOD
        if kind == "water":
            return self.WATER
        return self.EMPTY

    def _obs(self):
        x, y = self._pos
        view = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    view.append(0.0)
                else:
                    view.append(self._cell(x + dx, y + dy) / 3.0)
        obs = {"view": tuple(view),
               "satiety": self._satiety,
               "hydration": self._hydration,
               "steps_left": max(0.0, (self.MAX_STEPS - self._steps) / self.MAX_STEPS)}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "pos": self._pos, "satiety": self._satiety,
            "hydration": self._hydration,
            "resources": sorted((list(k), v) for k, v in self._resources.items()),
            "respawn_queue": self._respawn_queue, "steps": self._steps,
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }

    def state_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self._canonical_state()).encode("utf-8")).hexdigest()
