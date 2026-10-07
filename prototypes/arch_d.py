"""ARCHITECTURE D — lean causal core. The null hypothesis with a pulse.

Data flow (exactly as specified in research/architecture_candidates.md):

    perception
      -> single-level predictor -> prediction error
      -> episodic store: retrieve-k-similar, admit-if-surprising
      -> tiny workspace (K=4, plain top-k by bid, NO ignition theater)
      -> action: greedy on predicted value
      -> environment -> perception

Deliberately absent: PAD, narrative, phenomenal readout, binding math,
hierarchy, precision weighting, metacognition, motivation layer.

Every fancier architecture (A/B/C) must beat D by a preregistered margin
on the §35 battery, or its extra machinery is rejected. D itself is never
"the answer" — it is the bar.

Learning (§25): the predictor's weights change with experience and persist
across episodes; the episodic store grows by surprise-gated admission;
retrieval changes action scoring, so behavior changes with experience.

Known limitation (documented, not hidden): the store records IMMEDIATE
transition rewards, so multi-step credit assignment (delayed_reward) is
weak. A stronger episodic controller would store returns; that is a
hypothesis for a future architecture, not smuggled into the baseline.

~300 lines. Stdlib only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments", "baselines"))

from agent_base import BaselineAgent, TransitionLearner, flatten_obs, argmax_det  # noqa: E402
from predictor import LinearPredictor  # noqa: E402


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class ArchD(BaselineAgent, TransitionLearner):
    """Lean causal core. NAME = 'arch_d'."""

    NAME = "arch_d"
    VERSION = "1.0.0"

    LR = 0.05          # predictor learning rate
    K_RETRIEVE = 5     # episodic retrieval fan-out
    K_WORKSPACE = 4    # workspace capacity: plain top-k, no ignition
    STORE_CAP = 256    # episodic store capacity (FIFO eviction)
    SURPRISE_K = 1.0   # admit if error > ema + K*std
    EMA_ALPHA = 0.05   # recency weight for the surprise baseline
    RETRIEVAL_W = 0.5  # weight of retrieved-episode value vs predicted reward

    def _on_reset(self):
        self._tl_reset()
        if not hasattr(self, "_pred"):
            self._pred = None
            self._store = []       # list of [obs_vec, action, reward, error]
            self._ema_n = 0        # surprise baseline: EMA of squared error
            self._ema = 0.0
            self._ema_sq = 0.0
            self._last_surprise = 1.0
        self._workspace = []       # rebuilt every act(): [(bid, kind, payload)]
        self._dim = None

    # -- perception -> prediction ------------------------------------------------
    def _ensure_pred(self, obs):
        v = flatten_obs(obs)
        if self._pred is None:
            self._pred = LinearPredictor(len(v), self._n_actions,
                                         self.LR, self._rng)
        self._dim = len(v)  # always refresh: reset() clears it per episode
        return v

    # -- episodic store -----------------------------------------------------------
    def _retrieve(self, vec):
        """k most similar episodes by euclidean distance on obs vectors."""
        if not self._store:
            return []
        ranked = sorted(self._store, key=lambda e: _dist(vec, e[0]))
        return ranked[:self.K_RETRIEVE]

    def _admit(self, vec, action, reward, error):
        """Admit-if-surprising: error above EMA baseline + K * EMA std.

        The baseline is recency-weighted (EMA), not all-history: surprise is
        relative to CURRENT expectations. All-history statistics get
        dominated by the initial learning transient and then admit nothing.
        """
        if self._ema_n < 2:
            threshold = 0.0  # admit everything until the baseline exists
        else:
            var = max(0.0, self._ema_sq - self._ema ** 2)
            threshold = self._ema + self.SURPRISE_K * var ** 0.5
        if error > threshold:
            self._store.append([list(vec), action, reward, error])
            if len(self._store) > self.STORE_CAP:
                self._store.pop(0)  # FIFO eviction (documented, not learned)
            return True
        return False

    def _surprise_update(self, error):
        a = self.EMA_ALPHA
        self._ema_n += 1
        self._ema = (1 - a) * self._ema + a * error
        self._ema_sq = (1 - a) * self._ema_sq + a * error * error

    # -- learning hook (via TransitionLearner: every transition exactly once) ----
    def learn_transition(self, obs_vec, action, next_obs_vec, reward, done):
        error = self._pred.update(obs_vec, action, next_obs_vec, reward)
        self._surprise_update(error)
        self._last_surprise = error
        self._admit(obs_vec, action, reward, error)

    # -- workspace: plain top-k by bid --------------------------------------------
    def _build_workspace(self, vec, retrieved):
        items = [("percept", self._last_surprise, ("percept", tuple(vec)))]
        for ep in retrieved:
            sim = 1.0 / (1.0 + _dist(vec, ep[0]))
            bid = sim * (1.0 + abs(ep[2]))
            items.append(("episode", bid, ("episode", ep[1], ep[2])))
        items.sort(key=lambda t: t[1], reverse=True)
        # No ignition, no recurrence, no threshold theater: top-k wins.
        self._workspace = [(kind, bid, payload)
                           for kind, bid, payload in items[:self.K_WORKSPACE]]

    # -- act: greedy on predicted value --------------------------------------------
    def act(self, obs: dict) -> int:
        v = self._ensure_pred(obs)
        retrieved = self._retrieve(v)
        self._build_workspace(v, retrieved)
        by_action = {}
        for ep in retrieved:
            by_action.setdefault(ep[1], []).append(ep[2])
        scores = []
        for a in range(self._n_actions):
            pred_r = self._pred.predict(v, a)[1]
            mem_r = (sum(by_action[a]) / len(by_action[a])) if a in by_action else 0.0
            scores.append(pred_r + self.RETRIEVAL_W * mem_r)
        return argmax_det(scores)

    def update(self, obs: dict, action: int, reward: float,
               done: bool, info: dict) -> None:
        v = self._ensure_pred(obs)
        self._tl_update(v, action, reward, done, self._dim)

    # -- observability (§45): what won the workspace, and why ----------------------
    def workspace_contents(self):
        """[(kind, bid, payload)] — the current top-k shortlist."""
        return list(self._workspace)

    def memory_stats(self):
        var = max(0.0, self._ema_sq - self._ema ** 2)
        return {"store_size": len(self._store),
                "surprise_ema": self._ema, "surprise_std": var ** 0.5,
                "last_surprise": self._last_surprise}

    # -- continuity (§29) ------------------------------------------------------------
    def _extra_state(self):
        s = self._tl_state()
        s.update({
            "pred": None if self._pred is None else self._pred.get_state(),
            "store": [[list(e[0]), e[1], e[2], e[3]] for e in self._store],
            "surprise": [self._ema_n, self._ema, self._ema_sq],
            "last_surprise": self._last_surprise,
        })
        return s

    def _restore_extra(self, state):
        self._tl_restore(state)
        if state["pred"] is None:
            self._pred = None
        else:
            p = state["pred"]
            self._dim = p["d"]
            self._pred = LinearPredictor(p["d"], p["n"], p["lr"], self._rng)
            self._pred.set_state(p)
        self._store = [[tuple(e[0]), e[1], e[2], e[3]] for e in state["store"]]
        self._ema_n, self._ema, self._ema_sq = state["surprise"]
        self._last_surprise = state["last_surprise"]
        self._workspace = []
