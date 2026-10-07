"""Active inference action selection — Architecture B.

Select actions by minimizing an expected-free-energy proxy::

    G(a) = - lamV * Vhat(a) - lamIG * IGhat(a) + lamR * Riskhat(a)

  Vhat(a)  predicted reward (pragmatic value; action consequence model)
  IGhat(a) predicted precision-weighted L1 error magnitude for action a
           (epistemic value = expected information gain: the reducible
           error this action is expected to produce, estimated from the
           episodic store's per-action error history, falling back to a
           running per-action table)
  Riskhat(a) aleatoric floor: current mean channel noise (irreducible)

"Minimizing expected future error (information gain)": the IGhat term seeks
actions whose consequences are currently mispredicted in a *learnable* way
(high precision-weighted L1 error), while Vhat avoids reward loss and Risk
penalizes irreducible noise. Modes:

  active_inference  argmin_a G(a)
  greedy            argmax_a Vhat(a)
  random            uniform (seeded RNG)

The greedy and random modes exist so the active-inference label can be
killed: it must beat both on information-gain metrics (load-bearing
claim 4) or the label comes off.

Deterministic given the seed. Stdlib only.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List


class ActiveInferenceSelector:
    """Expected-error-minimizing action selection over discrete actions."""

    def __init__(self, n_actions: int, lamV: float = 1.0,
                 lamIG: float = 0.5, lamR: float = 0.10,
                 mode: str = "active_inference", seed: int = 0) -> None:
        if mode not in ("active_inference", "greedy", "random"):
            raise ValueError(f"unknown mode {mode!r}")
        self.n_actions = n_actions
        self.lamV = lamV
        self.lamIG = lamIG
        self.lamR = lamR
        self.mode = mode
        self._rng = random.Random(seed)
        # Running per-action mean |e1| table (fallback when memory is empty).
        self._err_sum = [0.0] * n_actions
        self._err_n = [0] * n_actions
        self.last_scores: List[float] = []

    def observe_error(self, action: int, e1_norm: float) -> None:
        self._err_sum[action] += float(e1_norm)
        self._err_n[action] += 1

    def table_error(self, action: int) -> float:
        n = self._err_n[action]
        if n == 0:
            total = sum(self._err_sum)
            tn = sum(self._err_n)
            return total / tn if tn else 0.5
        return self._err_sum[action] / n

    def select(self, predict_reward, predicted_error, risk: float) -> int:
        """Choose an action.

        predict_reward(a) -> float, predicted_error(a) -> float (both pure).
        risk is the scalar aleatoric floor.
        """
        if self.mode == "random":
            return self._rng.randrange(self.n_actions)
        scores = []
        for a in range(self.n_actions):
            if self.mode == "greedy":
                scores.append(-predict_reward(a))  # argmin form
            else:
                g = (-self.lamV * predict_reward(a)
                     - self.lamIG * predicted_error(a)
                     + self.lamR * risk)
                scores.append(g)
        self.last_scores = list(scores)
        best = min(range(self.n_actions), key=lambda a: scores[a])
        return best

    def snapshot(self) -> Dict:
        return {"n_actions": self.n_actions, "lamV": self.lamV,
                "lamIG": self.lamIG, "lamR": self.lamR, "mode": self.mode,
                "err_sum": self._err_sum, "err_n": self._err_n,
                "rng_state": self._rng.getstate()}

    def restore(self, state: Dict) -> None:
        if state["n_actions"] != self.n_actions:
            raise ValueError("selector action mismatch on restore")
        self.lamV = state["lamV"]; self.lamIG = state["lamIG"]
        self.lamR = state["lamR"]; self.mode = state["mode"]
        self._err_sum = state["err_sum"]; self._err_n = state["err_n"]
        self._rng.setstate(state["rng_state"])
