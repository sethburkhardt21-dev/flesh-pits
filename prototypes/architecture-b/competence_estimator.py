"""Competence estimator — EXP-FP-0070 variant A (additive prototype).

A genuine, causal, online competence estimator for Architecture B's §30
prediction domains. For each domain it emits, BEFORE the outcome is known:

  p_t    = P(event_t = 1 | causal history)   -- calibrated probability
  ehat_t = E[|error_t| | causal history]     -- expected error magnitude

Design (frozen in experiments/preregistration_EXP-FP-0070.json):
  p_t    = sigmoid(logit(p_climo,t) + w . z_t + b)     (online logistic)
  ehat_t = exp(w_e . z_t + b_e)                        (online log-linear)

p_climo,t is the sequential Laplace-smoothed running base rate of the event
over PAST outcomes only -- the identical baseline the estimator must beat.
With w=0/b=0 the estimator reduces EXACTLY to climatology, so every bit of
Brier improvement is earned by the learned feature weights (the anchor-only
kill arm proves this: learning disabled => Brier == Brier_climo to 1e-9).

Causality contract: predict() may use only (a) prediction-time fields of the
current tick (stated uncertainties, uhat -- known to the agent BEFORE the
outcome) and (b) quantities derived from ticks < t. observe() trains on tick
t's outcome AFTER the prediction is emitted (prequential). update_norm is
lagged one tick. No RNG anywhere; fully deterministic.

This module is ADDITIVE: it does not modify agent.py, predictions.py, or any
core file. The battery driver replays the agent's recorded PredictionLog
JSONL stream causally (prequential evaluation). Wiring into the live act()
path is future work -- SUCCESS here claims PROTOTYPE, never INTEGRATED.

CONSCIOUSNESS: UNRESOLVED. This is error-magnitude prediction machinery,
not a claim about experience.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


def _clamp01(p: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, p))


def _sgd_step(w: List[float], b: float, z: List[float], g: float,
              eta: float) -> float:
    """One normalized-SGD step: w -= eta*g*z/(1+||z||^2), b -= eta*g*b_scale.

    Normalization by (1+||z||^2) bounds the effective step and prevents
    divergence when z-scored features spike during regime shifts; the bias
    uses a gentler fixed scale. Deterministic, no RNG."""
    norm2 = 1.0 + sum(zi * zi for zi in z)
    s = eta * g / norm2
    for i in range(len(w)):
        w[i] -= s * z[i]
    return b - eta * g * 0.25


class _OnlineZ:
    """Online z-scorer (Welford). z=0 until variance exists."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def z(self, x: float) -> float:
        if self.n < 2:
            return 0.0
        var = self.m2 / (self.n - 1)
        if var <= 1e-12:
            return 0.0
        return (x - self.mean) / math.sqrt(var)

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)


class _ChannelState:
    """Causal error-channel history: fast/slow EMAs, surprise, lagged churn."""

    def __init__(self, alpha_fast: float = 0.15,
                 alpha_slow: float = 0.02) -> None:
        self.af = alpha_fast
        self.as_ = alpha_slow
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.ema_slow_sq: Optional[float] = None  # E[x^2] for surprise denom
        self.last: Optional[float] = None
        self.upd_ema: Optional[float] = None  # lagged update_norm EMA

    def observe(self, err: float, update_norm_lag: float) -> None:
        if self.ema_fast is None:
            self.ema_fast = err
            self.ema_slow = err
            self.ema_slow_sq = err * err
            self.upd_ema = update_norm_lag
        else:
            self.ema_fast += self.af * (err - self.ema_fast)
            self.ema_slow += self.as_ * (err - self.ema_slow)
            self.ema_slow_sq += self.as_ * (err * err - self.ema_slow_sq)
            self.upd_ema += self.af * (update_norm_lag - self.upd_ema)
        self.last = err

    def surprise_z(self) -> float:
        if self.last is None or self.ema_slow is None:
            return 0.0
        var = max(0.0, self.ema_slow_sq - self.ema_slow ** 2)
        return (self.last - self.ema_slow) / math.sqrt(var + 1e-9)

    def f_fast(self) -> float:
        return math.log1p(self.ema_fast) if self.ema_fast is not None else 0.0

    def f_slow(self) -> float:
        return math.log1p(self.ema_slow) if self.ema_slow is not None else 0.0

    def f_upd(self) -> float:
        return math.log1p(self.upd_ema) if self.upd_ema is not None else 0.0


class DomainEstimator:
    """One binary event + one error channel. Online logistic + log-linear."""

    def __init__(self, name: str, eta_p: float = 0.1,
                 eta_e: float = 0.05, learn: bool = True) -> None:
        self.name = name
        self.eta_p = eta_p
        self.eta_e = eta_e
        self.learn = learn
        self.chan = _ChannelState()
        self.zsc: List[_OnlineZ] = []  # grown lazily to feature dim
        # event base rate (Laplace): counts of PAST outcomes only
        self._a = 1.0
        self._n = 0
        self._event_ema: Optional[float] = None  # fast EMA of past events
        self.w: List[float] = []
        self.b = 0.0
        self.we: List[float] = []
        self.be = 0.0

    # -- causal state -----------------------------------------------------
    def climo(self) -> float:
        return self._a / (self._n + 2.0)

    def _features(self, stated_unc: float,
                  extra: List[float]) -> List[float]:
        # [log1p(EMA_fast|err|), log1p(EMA_slow|err|), z_surprise,
        #  logit(climo), log1p(stated_unc), EMA_fast(event_rate),
        #  lag1_update_norm_EMA] + extra
        ev = self._event_ema if self._event_ema is not None else 0.5
        feats = [self.chan.f_fast(), self.chan.f_slow(),
                 self.chan.surprise_z(), _logit(self.climo()),
                 math.log1p(max(0.0, stated_unc)), ev,
                 self.chan.f_upd()] + list(extra)
        return feats

    def _zscore(self, feats: List[float]) -> List[float]:
        while len(self.zsc) < len(feats):
            self.zsc.append(_OnlineZ())
        return [zs.z(x) for zs, x in zip(self.zsc, feats)]

    def _ensure_w(self, dim: int) -> None:
        while len(self.w) < dim:
            self.w.append(0.0)
            self.we.append(0.0)

    # -- online interface --------------------------------------------------
    def predict(self, stated_unc: float,
                extra: Optional[List[float]] = None) -> Dict[str, float]:
        feats = self._features(stated_unc, extra or [])
        z = self._zscore(feats)
        self._ensure_w(len(z))
        p = _clamp01(_sigmoid(_logit(self.climo())
                              + sum(a * b for a, b in zip(self.w, z))
                              + self.b))
        ehat = math.exp(max(-7.0, min(3.0,
            sum(a * b for a, b in zip(self.we, z)) + self.be)))
        return {"p": p, "ehat": ehat,
                "uncertainty_anchor": 1.0 - p, "climo": self.climo()}

    def observe(self, err: float, event: int, stated_unc: float,
                update_norm_lag: float,
                extra: Optional[List[float]] = None,
                feat_err: Optional[float] = None) -> None:
        """feat_err: when given (mode C), the error-CHANNEL features
        (EMAs, surprise) and the ehat head track feat_err (the predictable
        rest component) instead of err, while the p head still trains on
        the preregistered (err, event). All causal: features used for
        training are from history < current tick."""
        feats = self._features(stated_unc, extra or [])
        z = self._zscore(feats)
        self._ensure_w(len(z))
        ferr = feat_err if feat_err is not None else err
        if self.learn:
            p = _sigmoid(_logit(self.climo())
                         + sum(a * b for a, b in zip(self.w, z)) + self.b)
            g = p - event
            self.b = _sgd_step(self.w, self.b, z, g, self.eta_p)
            le = math.log(max(1e-6, ferr))
            pe = sum(a * b for a, b in zip(self.we, z)) + self.be
            ge = pe - le
            self.be = _sgd_step(self.we, self.be, z, ge, self.eta_e)
        # advance causal state AFTER training on this tick
        for zs, x in zip(self.zsc, feats):
            zs.update(x)
        self.chan.observe(ferr, update_norm_lag)
        self._a += event
        self._n += 1
        if self._event_ema is None:
            self._event_ema = float(event)
        else:
            self._event_ema += 0.15 * (event - self._event_ema)

    def reset(self) -> None:
        self.__init__(self.name, self.eta_p, self.eta_e, self.learn)


class CompetenceEstimator:
    """Four-domain competence estimator (EXP-FP-0070 variant A).

    Domains mirror EXP-FP-CALIB-01's binary events; thresholds are passed in
    (frozen from history, never fit on battery data).
    """

    DOMAINS = ("next_obs", "action_consequence",
               "competence_failure", "retrieval_usefulness")

    def __init__(self, eps_obs: float, eps_rw: float, tau_fail: float,
                 learn: bool = True, feature_mode: str = "A") -> None:
        self.eps_obs = eps_obs
        self.eps_rw = eps_rw
        self.tau_fail = tau_fail
        self.feature_mode = feature_mode
        self.est = {d: DomainEstimator(d, learn=learn)
                    for d in self.DOMAINS}

    @staticmethod
    def _novelty_extras(ctx: Dict[str, float]) -> List[float]:
        """Prediction-time novelty/state signals (EXP-FP-0071 variant B).

        All are known to the agent BEFORE the tick's outcome (captured from
        _pending at _open_tick time / e1_ema from the previous close).
        Missing keys default to neutral zeros (mode-A replays)."""
        return [float(ctx.get("mean_similarity", 0.0)),
                float(ctx.get("n_nbrs", 0)) / 5.0,
                1.0 if ctx.get("retrieval_used") else 0.0,
                math.log1p(max(0.0, float(ctx.get("e1_ema", 0.0))))]

    def _extras(self, ctx: Dict[str, float],
                base: Optional[List[float]] = None) -> List[float]:
        base = base or []
        if self.feature_mode in ("B", "C", "D"):
            return base + self._novelty_extras(ctx)
        return base

    def reset(self) -> None:
        for e in self.est.values():
            e.reset()

    def predict_tick(self, ctx: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """ctx: prediction-time fields of this tick:
        ounc, runc, uunc, uhat (all known BEFORE the outcome); mode B adds
        mean_similarity, n_nbrs, retrieval_used, e1_ema from the sidecar."""
        nov = self._extras(ctx)
        return {
            "next_obs": self.est["next_obs"].predict(ctx["ounc"],
                                                     extra=nov),
            "action_consequence": self.est["action_consequence"].predict(
                ctx["runc"], extra=nov),
            "competence_failure": self.est["competence_failure"].predict(
                ctx["ounc"], extra=nov),
            "retrieval_usefulness": self.est["retrieval_usefulness"].predict(
                ctx["uunc"], extra=self._extras(ctx, [ctx["uhat"]])),
        }

    def observe_tick(self, ctx: Dict[str, float], out: Dict[str, float],
                     update_norm_lag: float) -> None:
        """out: tick outcomes -- err_next, err_rw, benefit (signed).

        Mode C: D1/D3's error-channel features and ehat head track rest_err
        (mean |signed_error| over obs dims 2-5, excluding the pure-RNG cue
        dims 0-1), computed here from the tick's recorded signed_error --
        used ONLY for post-prediction training, never for the prediction
        itself. The p heads always train on the preregistered (err, event).
        """
        nov = self._extras(ctx)
        if self.feature_mode == "D":
            # EXP-FP-0100: estimand re-scoping. The pure-RNG cue channels
            # (obs dims 0-1) are EXCLUDED from the estimand: D1/D3's p heads
            # train on REST-error events (competence = predictable error,
            # not total error). The caller passes the re-scoped thresholds
            # as eps_obs (EPS_OBS_REST) and tau_fail (TAU_FAIL_REST);
            # eps_rw (D2) and D4 are unchanged. Driver must supply
            # ctx["signed_error"]; missing it is a hard error, never a
            # silent fallback (fail closed, not fail soft).
            if "signed_error" not in ctx:
                raise ValueError(
                    "feature_mode='D' requires ctx['signed_error']")
            se = ctx["signed_error"]
            rest_err = sum(abs(x) for x in se[2:6]) / 4.0
            self.est["next_obs"].observe(
                rest_err, 1 if rest_err <= self.eps_obs else 0,
                ctx["ounc"], update_norm_lag, extra=nov, feat_err=rest_err)
            self.est["action_consequence"].observe(
                out["err_rw"], 1 if out["err_rw"] <= self.eps_rw else 0,
                ctx["runc"], update_norm_lag, extra=nov)
            self.est["competence_failure"].observe(
                rest_err, 1 if rest_err > self.tau_fail else 0,
                ctx["ounc"], update_norm_lag, extra=nov, feat_err=rest_err)
            self.est["retrieval_usefulness"].observe(
                abs(out["benefit"]), 1 if out["benefit"] > 0 else 0,
                ctx["uunc"], update_norm_lag,
                extra=self._extras(ctx, [ctx["uhat"]]))
            return
        rest_err = None
        if self.feature_mode == "C" and "signed_error" in ctx:
            se = ctx["signed_error"]
            rest_err = sum(abs(x) for x in se[2:6]) / 4.0
        self.est["next_obs"].observe(
            out["err_next"], 1 if out["err_next"] <= self.eps_obs else 0,
            ctx["ounc"], update_norm_lag, extra=nov, feat_err=rest_err)
        self.est["action_consequence"].observe(
            out["err_rw"], 1 if out["err_rw"] <= self.eps_rw else 0,
            ctx["runc"], update_norm_lag, extra=nov)
        self.est["competence_failure"].observe(
            out["err_next"], 1 if out["err_next"] > self.tau_fail else 0,
            ctx["ounc"], update_norm_lag, extra=nov, feat_err=rest_err)
        self.est["retrieval_usefulness"].observe(
            abs(out["benefit"]), 1 if out["benefit"] > 0 else 0,
            ctx["uunc"], update_norm_lag,
            extra=self._extras(ctx, [ctx["uhat"]]))
