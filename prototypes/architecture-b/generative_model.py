"""Hierarchical generative model — Architecture B core (v2: context experts).

Two levels (L0, L1). Belief states PERSIST across ticks (this is the
rewrite: the old free_energy_relaxation.py did one-shot relaxation with
no carried belief).

L0 (fast): linear maps W (observations) and w_r/b_r (reward).
L1 (slow): CONTEXT-INDEXED EXPERTS. mu1 = {D_ctx, R_ctx} tables, where the
  context key is the tuple of categorical observation values (generic
  extraction — for changing_rule this is the cue; envs without
  categoricals get a single context). D_ctx is a linear residual expert;
  R_ctx is a per-action reward table. Both persist across ticks and are
  updated by precision-weighted prediction errors.

  (v1 used a mixture-of-experts mu1 vector with input-free gating. It had
  a fatal cold-start: V and mu1 needed each other to get gradient, so the
  interaction never learned. The context-expert redesign gives clean
  credit assignment — only the active context's expert updates — and is
  recorded in research/negative_results.md as NR-B-001.)

Per tick::

    ctx      = categorical values of obs_t
    predict: xhat = W.f + D_ctx.f ; rhat = w_r.f + b_r + R_ctx[a]
             f = [mu0, onehot(a)]   (predictions condition on the belief)
    observe: x_{t+1} arrives
    errors:  e0 = x_{t+1} - xhat            (sensory prediction error)
             e1 = e0 - D_ctx.f              (residual L1 failed to explain)
             rerr = r - rhat
    updates: dW      = eta0 * (pi0 * e0) (x) f
             dD_ctx  = etaD * (pi1 * e1) (x) f     (active context only)
             dw_r/db = eta_r * piR * rerr * f
             dR_ctx[a] = etaR * piR * rerr         (taken action only)
             mu0 <- xhat + e0 / (1 + pi0)          (Kalman-like posterior)

Precision pi0/pi1/piR comes from PrecisionEstimator (error statistics),
never hand-set gains. All learning rates are settable to 0 (K1 freeze).
lesion_l1=True disables the D_ctx/R_ctx paths and their updates (K3).

Deterministic given the seed. Stdlib only. No identity strings.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from precision import PrecisionEstimator


def _zeros(n: int) -> List[float]:
    return [0.0] * n


def _matvec(m: List[List[float]], v: List[float]) -> List[float]:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


class HierarchicalGenerativeModel:
    """Two-level predictive-coding world model with persistent beliefs."""

    def __init__(self, obs_dim: int, n_actions: int,
                 cat_slices: List[Tuple[int, int]] = (),
                 eta0: float = 0.005, etaD: float = 0.002,
                 eta_r: float = 0.05, etaR: float = 0.10,
                 max_contexts: int = 32,
                 precision_window: int = 50, uniform_precision: bool = False,
                 seed: int = 0) -> None:
        if obs_dim < 1 or n_actions < 1:
            raise ValueError("dims must be >= 1")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.feat_dim = obs_dim + n_actions
        self.cat_slices = list(cat_slices)
        self.eta0 = eta0
        self.etaD = etaD
        self.eta_r = eta_r
        self.etaR = etaR
        self.max_contexts = max_contexts
        self.seed = seed
        self._rng = random.Random(seed)

        scale = 0.05
        # L0 fast weights.
        self.W: List[List[float]] = [
            [self._rng.uniform(-scale, scale) for _ in range(self.feat_dim)]
            for _ in range(obs_dim)]
        self.w_r: List[float] = [0.0] * self.feat_dim
        self.b_r = 0.0

        # L1: context-indexed experts (the persistent slow belief mu1).
        self.ctx_D: Dict[Tuple[int, ...], List[List[float]]] = {}
        self.ctx_R: Dict[Tuple[int, ...], List[float]] = {}

        # L0 posterior state belief (persists across ticks).
        self.mu0: List[float] = _zeros(obs_dim)

        self.prec0 = PrecisionEstimator(obs_dim, window=precision_window,
                                        uniform=uniform_precision)
        self.prec1 = PrecisionEstimator(obs_dim, window=precision_window,
                                        uniform=uniform_precision)
        self.precR = PrecisionEstimator(1, window=precision_window,
                                        uniform=uniform_precision)

        self._r_err_mean = 0.0
        self._r_err_m2 = 0.0
        self._r_err_n = 0

        self.last_e0: List[float] = _zeros(obs_dim)
        self.last_e1: List[float] = _zeros(obs_dim)
        self.last_pi0: List[float] = [1.0] * obs_dim
        self.last_pi1: List[float] = [1.0] * obs_dim
        self.last_update_norm = 0.0

    # -- context ----------------------------------------------------------
    def context_key(self, obs_vec: List[float]) -> Tuple[int, ...]:
        """Discrete context = argmax of each categorical slice."""
        key = []
        for (s, e) in self.cat_slices:
            seg = obs_vec[s:e]
            key.append(max(range(len(seg)), key=lambda i: seg[i]))
        return tuple(key)

    def _experts(self, ctx: Tuple[int, ...]):
        """Get (creating if needed) the experts for a context."""
        if ctx not in self.ctx_D:
            if len(self.ctx_D) >= self.max_contexts:
                # Cap: reuse the least-recently-created context's experts
                # (documented; keeps memory bounded on wild spaces).
                ctx = next(iter(self.ctx_D))
            else:
                scale = 0.02
                self.ctx_D[ctx] = [
                    [self._rng.uniform(-scale, scale)
                     for _ in range(self.feat_dim)]
                    for _ in range(self.obs_dim)]
                self.ctx_R[ctx] = [0.0] * self.n_actions
        return self.ctx_D[ctx], self.ctx_R[ctx]

    @property
    def n_contexts(self) -> int:
        return len(self.ctx_D)

    # -- features -----------------------------------------------------------
    def feat(self, state_vec: List[float], action: int) -> List[float]:
        if len(state_vec) != self.obs_dim:
            raise ValueError("state dim mismatch")
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of range")
        onehot = [1.0 if a == action else 0.0 for a in range(self.n_actions)]
        return list(state_vec) + onehot

    # -- prediction -----------------------------------------------------------
    def topdown(self, feat: List[float],
                ctx: Tuple[int, ...]) -> List[float]:
        """L1 -> L0 prediction of the structured residual."""
        D, _ = self._experts(ctx)
        return _matvec(D, feat)

    def predict_next(self, state_vec: List[float], action: int,
                     ctx: Tuple[int, ...]) -> List[float]:
        """Full prediction. state_vec is the STATE BELIEF (mu0): predictions
        flow from beliefs; raw sensation enters only via prediction error."""
        f = self.feat(state_vec, action)
        base = _matvec(self.W, f)
        td = self.topdown(f, ctx)
        return [base[c] + td[c] for c in range(self.obs_dim)]

    def predict_reward(self, state_vec: List[float], action: int,
                       ctx: Tuple[int, ...]) -> float:
        f = self.feat(state_vec, action)
        _, R = self._experts(ctx)
        return (sum(w * x for w, x in zip(self.w_r, f)) + self.b_r
                + R[action])

    # -- observation + belief update -------------------------------------------
    def observe(self, state_vec: List[float], action: int,
                ctx: Tuple[int, ...], obs_next: List[float], reward: float,
                lesion_l1: bool = False) -> Dict:
        """Close one tick. lesion_l1 disables the L1 experts (K3)."""
        if len(obs_next) != self.obs_dim:
            raise ValueError("obs_next dim mismatch")
        f = self.feat(state_vec, action)
        base = _matvec(self.W, f)
        D, R = self._experts(ctx)
        td = [0.0] * self.obs_dim if lesion_l1 else _matvec(D, f)
        r_ctx = 0.0 if lesion_l1 else R[action]
        xhat = [base[c] + td[c] for c in range(self.obs_dim)]
        rhat = (sum(w * x for w, x in zip(self.w_r, f)) + self.b_r + r_ctx)

        e0 = [obs_next[c] - xhat[c] for c in range(self.obs_dim)]
        e1 = [e0[c] - td[c] for c in range(self.obs_dim)]
        rerr = reward - rhat

        pi0 = self.prec0.observe(e0)
        pi1 = self.prec1.observe(e1)
        piR = self.precR.observe([rerr])[0]
        werr0 = [pi0[c] * e0[c] for c in range(self.obs_dim)]
        werr1 = [pi1[c] * e1[c] for c in range(self.obs_dim)]

        upd_sq = 0.0
        # L0: dW = eta0 * (pi0*e0) (x) feat
        for c in range(self.obs_dim):
            for j in range(self.feat_dim):
                d = self.eta0 * werr0[c] * f[j]
                self.W[c][j] += d
                upd_sq += d * d
        # L0 reward head.
        for j in range(self.feat_dim):
            d = self.eta_r * piR * rerr * f[j]
            self.w_r[j] += d
            upd_sq += d * d
        self.b_r += self.eta_r * piR * rerr
        # L1 experts (active context only; skipped under lesion).
        if not lesion_l1:
            for c in range(self.obs_dim):
                for j in range(self.feat_dim):
                    d = self.etaD * werr1[c] * f[j]
                    D[c][j] += d
                    upd_sq += d * d
            dR = self.etaR * piR * rerr
            R[action] += dR
            upd_sq += dR * dR

        # Posterior state belief (Kalman-like, bounded).
        self.mu0 = [xhat[c] + e0[c] / (1.0 + pi0[c])
                    for c in range(self.obs_dim)]

        self._r_err_n += 1
        d = rerr - self._r_err_mean
        self._r_err_mean += d / self._r_err_n
        self._r_err_m2 += d * (rerr - self._r_err_mean)

        self.last_e0 = list(e0)
        self.last_e1 = list(e1)
        self.last_pi0 = list(pi0)
        self.last_pi1 = list(pi1)
        self.last_update_norm = math.sqrt(upd_sq)
        return {
            "e0": list(e0), "e1": list(e1),
            "pi0": list(pi0), "pi1": list(pi1), "piR": piR,
            "xhat": xhat, "rhat": rhat, "rerr": rerr,
            "ctx": ctx, "n_contexts": len(self.ctx_D),
            "update_norm": self.last_update_norm,
        }

    def reward_confidence(self) -> Tuple[float, float]:
        if self._r_err_n < 2:
            return 0.0, 1.0
        var = self._r_err_m2 / (self._r_err_n - 1)
        unc = math.sqrt(var + 1e-9)
        return 1.0 / (1.0 + unc), unc

    def obs_confidence(self) -> Tuple[float, float]:
        var = self.prec0.channel_variance()
        unc = math.sqrt(sum(var) / max(1, len(var)) + 1e-9)
        return 1.0 / (1.0 + unc), unc

    # -- persistence -----------------------------------------------------------
    def snapshot(self) -> Dict:
        return {
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "cat_slices": self.cat_slices,
            "eta0": self.eta0, "etaD": self.etaD,
            "eta_r": self.eta_r, "etaR": self.etaR,
            "max_contexts": self.max_contexts, "seed": self.seed,
            "W": self.W, "w_r": self.w_r, "b_r": self.b_r,
            "ctx_D": {",".join(map(str, k)): v
                      for k, v in self.ctx_D.items()},
            "ctx_R": {",".join(map(str, k)): v
                      for k, v in self.ctx_R.items()},
            "mu0": self.mu0,
            "prec0": self.prec0.snapshot(), "prec1": self.prec1.snapshot(),
            "precR": self.precR.snapshot(),
            "r_err": [self._r_err_mean, self._r_err_m2, self._r_err_n],
        }

    def restore(self, state: Dict) -> None:
        for key in ("obs_dim", "n_actions"):
            if state[key] != getattr(self, key):
                raise ValueError(f"model dim mismatch on restore: {key}")
        self.cat_slices = [tuple(s) for s in state["cat_slices"]]
        self.eta0 = state["eta0"]; self.etaD = state["etaD"]
        self.eta_r = state["eta_r"]; self.etaR = state["etaR"]
        self.max_contexts = state["max_contexts"]; self.seed = state["seed"]
        self.W = state["W"]; self.w_r = state["w_r"]; self.b_r = state["b_r"]
        self.ctx_D = {tuple(map(int, k.split(","))): v
                      for k, v in state["ctx_D"].items()}
        self.ctx_R = {tuple(map(int, k.split(","))): v
                      for k, v in state["ctx_R"].items()}
        self.mu0 = state["mu0"]
        self.prec0.restore(state["prec0"]); self.prec1.restore(state["prec1"])
        self.precR.restore(state["precR"])
        self._r_err_mean, self._r_err_m2, self._r_err_n = state["r_err"]
