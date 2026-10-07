"""Online linear predictor shared by predictor-based baselines. Stdlib only.

Maps x = [obs_vec; onehot(action); 1.0(bias)] -> (predicted next obs_vec,
predicted immediate reward) with SGD on squared error.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LinearPredictor:
    def __init__(self, obs_dim, n_actions, lr, rng, init_scale=0.1):
        self.d = obs_dim
        self.n = n_actions
        self.lr = lr
        din = obs_dim + n_actions + 1
        self.W = [[rng.uniform(-init_scale, init_scale) for _ in range(obs_dim)]
                  for _ in range(din)]
        self.w = [rng.uniform(-init_scale, init_scale) for _ in range(din)]

    def _x(self, obs_vec, action):
        return (list(obs_vec)
                + [1.0 if i == action else 0.0 for i in range(self.n)]
                + [1.0])

    def predict(self, obs_vec, action):
        x = self._x(obs_vec, action)
        nxt = [sum(x[i] * self.W[i][j] for i in range(len(x)))
               for j in range(self.d)]
        rew = sum(x[i] * self.w[i] for i in range(len(x)))
        return nxt, rew

    def update(self, obs_vec, action, next_obs_vec, reward):
        """One SGD step. Returns total squared error (surprise signal)."""
        x = self._x(obs_vec, action)
        nxt_pred, rew_pred = self.predict(obs_vec, action)
        eo = [nxt_pred[j] - next_obs_vec[j] for j in range(self.d)]
        er = rew_pred - reward
        lr = self.lr
        for i, xi in enumerate(x):
            if xi == 0.0:
                continue
            self.w[i] -= lr * er * xi
            Wi = self.W[i]
            for j in range(self.d):
                Wi[j] -= lr * eo[j] * xi
        return sum(e * e for e in eo) + er * er

    # -- continuity ------------------------------------------------------
    def get_state(self):
        return {"d": self.d, "n": self.n, "lr": self.lr,
                "W": self.W, "w": self.w}

    def set_state(self, s):
        assert (s["d"], s["n"]) == (self.d, self.n)
        self.lr = s["lr"]
        self.W = [list(row) for row in s["W"]]
        self.w = list(s["w"])
