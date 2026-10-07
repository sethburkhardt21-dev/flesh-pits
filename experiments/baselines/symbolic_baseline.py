"""Deterministic symbolic baseline — no LLM, no learning, no randomness (§39).

Hand-coded heuristics per environment family, selected at runtime from the
OBSERVATION KEY SIGNATURE (design-time knowledge of the documented channel
layouts — never from `info`, which is not read anywhere in this file).

Families (by obs keys):
  grid_world     {pos_x, pos_y, goal_dx, goal_dy, view, steps_left}
                 -> BFS on the mapped known grid toward the goal
                    (unknown = assumed free, known hazards = blocked).
  pomaze         {wall_n, wall_s, wall_e, wall_w, beacon, steps_left}
                 -> dead-reckoned mapping + BFS to nearest frontier
                    (nearest cell adjacent to unknown space).
  changing_rule  {cue, last_reward, last_action}
                 -> win-stay / lose-shift per cue.
  delayed_reward {pos, branch, steps_left}
                 -> always branch_a at t=0, then forward.
  resource_world {view, satiety, hydration, steps_left}
                 -> greedy move onto visible needed resource, else deterministic
                    wall-following coverage (fixed turn order, bump = view
                    unchanged after a move action).

Fully deterministic: the episode seed is ignored (kept only for interface
conformance). This is the "classical AI" bar: any learning architecture
should eventually beat hand-written heuristics, or the task is too easy to
prove learning mattered.
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_base import BaselineAgent, flatten_obs  # noqa: E402

DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W


def bfs_path(is_blocked, start, is_target, max_expand=10000):
    """BFS over grid cells. is_blocked(p)->bool, is_target(p)->bool.
    Returns list of cells from start to target (inclusive), or None."""
    if is_target(start):
        return [start]
    prev = {start: None}
    dq = deque([start])
    expanded = 0
    while dq and expanded < max_expand:
        cur = dq.popleft()
        expanded += 1
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in prev or is_blocked(nb):
                continue
            prev[nb] = cur
            if is_target(nb):
                path = [nb]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            dq.append(nb)
    return None


def step_to_action(frm, to, n_actions):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    idx = DIRS.index((dx, dy))
    assert idx < n_actions
    return idx


class SymbolicBaselineAgent(BaselineAgent):
    NAME = "symbolic"

    def _on_reset(self):
        self._family = None
        self._map = {}        # grid/pomaze: pos -> sensed cell info
        self._pos = (0, 0)    # dead-reckoned (pomaze) / decoded (grid)
        self._wsls = {0: 0, 1: 0}  # changing_rule: last action per cue
        self._last_view = None     # resource_world: bump detection
        self._last_action = 0

    # -- family detection ------------------------------------------------
    def _detect(self, obs):
        keys = set(obs.keys())
        if "cue" in keys:
            return "changing_rule"
        if "branch" in keys:
            return "delayed_reward"
        if "satiety" in keys:
            return "resource_world"
        if "wall_n" in keys:
            return "pomaze"
        return "grid_world"

    def act(self, obs: dict) -> int:
        if self._family is None:
            self._family = self._detect(obs)
        return {"grid_world": self._act_grid,
                "pomaze": self._act_pomaze,
                "changing_rule": self._act_wsls,
                "delayed_reward": self._act_delayed,
                "resource_world": self._act_resource}[self._family](obs)

    def update(self, obs, action, reward, done, info) -> None:
        if self._family == "changing_rule" and reward is not None:
            cue = obs["cue"]
            if reward == 0.0:
                self._wsls[cue] = 1 - self._wsls[cue]  # lose: shift
            # win: stay (no change)

    # -- grid_world: BFS to goal on known map ----------------------------
    def _act_grid(self, obs):
        gx = round(obs["pos_x"] * 7 + obs["goal_dx"] * 7)
        gy = round(obs["pos_y"] * 7 + obs["goal_dy"] * 7)
        px, py = round(obs["pos_x"] * 7), round(obs["pos_y"] * 7)
        self._pos = (px, py)
        goal = (gx, gy)
        view = obs["view"]
        for j, dy in enumerate((-1, 0, 1)):
            for i, dx in enumerate((-1, 0, 1)):
                code = view[j * 3 + i]
                if dx == 0 and dy == 0:
                    continue
                self._map[(px + dx, py + dy)] = code

        def blocked(p):
            x, y = p
            if not (0 <= x < 8 and 0 <= y < 8):
                return True
            c = self._map.get(p)
            if c is None:
                return False  # unknown = assumed free (optimistic)
            # wall (1/3) and hazard (1.0) blocked; goal (2/3) traversable
            return abs(c - 1.0 / 3.0) < 1e-9 or c >= 1.0 - 1e-9

        path = bfs_path(blocked, (px, py), lambda p: p == goal)
        if path and len(path) > 1:
            return step_to_action((px, py), path[1], self._n_actions)
        # fallback: greedy step reducing Manhattan distance to goal
        best, best_d = 0, abs(gx - px) + abs(gy - py)
        for a, (dx, dy) in enumerate(DIRS):
            if a >= self._n_actions:
                break
            d = abs(gx - (px + dx)) + abs(gy - (py + dy))
            if d < best_d and not blocked((px + dx, py + dy)):
                best, best_d = a, d
        return best

    # -- pomaze: frontier exploration ------------------------------------
    # Map values: 4-tuple of wall bools (fully sensed open cell),
    #             None (known open, walls unsensed),
    #             "wall" (known wall cell),
    #             missing (truly unknown).
    def _act_pomaze(self, obs):
        x, y = self._pos
        walls = (obs["wall_n"] > 0.5, obs["wall_s"] > 0.5,
                 obs["wall_e"] > 0.5, obs["wall_w"] > 0.5)
        self._map[(x, y)] = walls
        for a, (dx, dy) in enumerate(DIRS):
            nb = (x + dx, y + dy)
            if nb not in self._map:
                # Sensed adjacency: a walled neighbor is KNOWN wall, not
                # unknown frontier. (Missing this caused infinite oscillation
                # against out-of-sight walls.)
                self._map[nb] = "wall" if walls[a] else None

        def is_frontier(p):
            for dx, dy in DIRS:
                if (p[0] + dx, p[1] + dy) not in self._map:
                    return True
            return False

        # BFS to nearest frontier cell (a reachable cell adjacent to unknown
        # space). Start cell excluded: it was just fully sensed.

        # BFS to nearest frontier cell (a cell adjacent to unknown space).

        # BFS to nearest frontier cell; then step toward it.
        prev = {(x, y): None}
        dq = deque([(x, y)])
        target_path = None
        while dq and target_path is None:
            cur = dq.popleft()
            if cur != (x, y) and is_frontier(cur):
                path = [cur]
                while path[-1] != (x, y):
                    path.append(prev[path[-1]])
                target_path = path[::-1]
                break
            for dx, dy in DIRS:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in prev or self._map.get(nb) == "wall":
                    continue
                # can we step cur -> nb? need no wall between (sensed at cur
                # if cur fully sensed, else optimistic)
                cur_walls = self._map.get(cur)
                if isinstance(cur_walls, tuple):
                    a = DIRS.index((dx, dy))
                    if cur_walls[a]:
                        continue
                prev[nb] = cur
                dq.append(nb)

        if target_path and len(target_path) > 1:
            nxt = target_path[1]
        else:
            # Fully explored (should not happen before the goal is found):
            # first open direction in fixed order.
            nxt = None
            for a, (dx, dy) in enumerate(DIRS):
                if not walls[a]:
                    nxt = (x + dx, y + dy)
                    break
            if nxt is None:
                return 0
        action = step_to_action((x, y), nxt, self._n_actions)
        # dead reckoning: move succeeds iff no wall in that direction
        if not walls[action]:
            self._pos = nxt
        return action

    # -- changing_rule: win-stay / lose-shift -----------------------------
    def _act_wsls(self, obs):
        return self._wsls[obs["cue"]]

    # -- delayed_reward: fixed branch_a, then forward ---------------------
    def _act_delayed(self, obs):
        if obs["branch"] == 2:  # none chosen yet
            return 0  # branch_a
        return 2  # forward

    # -- resource_world: greedy-on-visible + wall following ----------------
    def _act_resource(self, obs):
        need_food = obs["satiety"] <= obs["hydration"]
        want = 2.0 / 3.0 if need_food else 1.0  # food code/3, water code/3
        view = obs["view"]
        # adjacent (incl. diagonal) resource of the needed kind?
        best = None
        for j, dy in enumerate((-1, 0, 1)):
            for i, dx in enumerate((-1, 0, 1)):
                if dx == 0 and dy == 0:
                    continue
                if abs(view[j * 3 + i] - want) < 1e-9:
                    best = (dx, dy)
                    break
            if best:
                break
        if best:
            dx, dy = best
            # step horizontally first, then vertically
            if dx != 0:
                return 2 if dx > 0 else 3  # east / west
            return 1 if dy > 0 else 0      # south / north
        # coverage: continue; on bump (view unchanged) turn right (fixed order)
        cur_view = tuple(view)
        if self._last_view is not None and cur_view == self._last_view:
            self._last_action = (self._last_action + 1) % 4
        self._last_view = cur_view
        return self._last_action

    # -- continuity ---------------------------------------------------------
    def _extra_state(self):
        return {"family": self._family,
                "map": {f"{k[0]},{k[1]}": v for k, v in self._map.items()},
                "pos": list(self._pos), "wsls": self._wsls,
                "last_action": self._last_action}

    def _restore_extra(self, state):
        self._family = state["family"]
        self._map = {}
        for k, v in state["map"].items():
            x, y = k.split(",")
            self._map[(int(x), int(y))] = tuple(v) if isinstance(v, list) else v
        self._pos = tuple(state["pos"])
        self._wsls = {int(k): v for k, v in state["wsls"].items()}
        self._last_action = state["last_action"]
        self._last_view = None
