"""Prediction targets (§30) — Architecture B.

Three targets, every tick:
  1. next observation        (L0 generative prediction, memory-corrected)
  2. action consequence       (predicted reward of the chosen action)
  3. retrieval usefulness     (predicted error-reduction from memory use)

Each tick records: prediction, confidence, uncertainty, actual,
signed error, absolute error, calibration, and the subsequent model update.
Records stream to JSONL (one line per tick per target); aggregate
calibration summaries go into experiment receipts.

Calibration: for each target we bin predictions by stated confidence and
compare mean |z|-style normalized error inside each bin. A well-calibrated
predictor has normalized error ~ flat across bins; we report the
calibration slope and mean absolute calibration error.

The retrieval-usefulness predictor is itself learned: uhat = w_u . qfeat,
qfeat = [mean_similarity, n_neighbours, current_uncertainty, e1_norm_ema],
trained on measured benefit = |e0_without| - |e0_with| whenever retrieval
is applied. This closes the §30 loop: the model predicts whether consulting
memory will help, then measures whether it did.

Deterministic. Stdlib only.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional


class CalibrationTracker:
    """Binned calibration: confidence bins vs normalized absolute error."""

    def __init__(self, n_bins: int = 5) -> None:
        self.n_bins = n_bins
        self._bins: List[List[float]] = [[] for _ in range(n_bins)]

    def observe(self, confidence: float, abs_err: float,
                uncertainty: float) -> None:
        """Record one (confidence, error) pair. Error normalized by stated
        uncertainty (z-like); a calibrated predictor keeps this ~flat."""
        b = min(self.n_bins - 1, max(0, int(confidence * self.n_bins)))
        znorm = abs_err / (uncertainty + 1e-9)
        self._bins[b].append(znorm)

    def summary(self) -> Dict:
        means = []
        for b in self._bins:
            means.append(sum(b) / len(b) if b else None)
        filled = [(i, m) for i, m in enumerate(means) if m is not None]
        slope = None
        if len(filled) >= 2:
            xs = [i for i, _ in filled]
            ys = [m for _, m in filled]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            if den > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        n = sum(len(b) for b in self._bins)
        return {"bin_means": means, "slope": slope, "n": n}


class UsefulnessPredictor:
    """Learned predictor of retrieval benefit (target 3)."""

    def __init__(self, eta: float = 0.10) -> None:
        self.eta = eta
        self.w = [0.0] * 4  # [mean_sim, n_nbrs/5, uncertainty, e1_ema]
        self.b = 0.0
        self._err_mean = 0.0
        self._err_m2 = 0.0
        self._n = 0

    def features(self, mean_sim: float, n_nbrs: int,
                 uncertainty: float, e1_ema: float) -> List[float]:
        return [float(mean_sim), n_nbrs / 5.0,
                float(uncertainty), float(e1_ema)]

    def predict(self, qfeat: List[float]) -> float:
        return sum(w * x for w, x in zip(self.w, qfeat)) + self.b

    def confidence(self) -> tuple:
        if self._n < 2:
            return 0.0, 1.0
        var = self._err_m2 / (self._n - 1)
        unc = math.sqrt(var + 1e-9)
        return 1.0 / (1.0 + unc), unc

    def update(self, qfeat: List[float], actual_benefit: float) -> Dict:
        pred = self.predict(qfeat)
        err = actual_benefit - pred
        for i in range(len(self.w)):
            self.w[i] += self.eta * err * qfeat[i]
        self.b += self.eta * err
        self._n += 1
        d = err - self._err_mean
        self._err_mean += d / self._n
        self._err_m2 += d * (err - self._err_mean)
        conf, unc = self.confidence()
        return {"prediction": pred, "actual": actual_benefit,
                "signed_error": err, "abs_error": abs(err),
                "confidence": conf, "uncertainty": unc,
                "update_norm": abs(self.eta * err) *
                math.sqrt(sum(x * x for x in qfeat) + 1.0)}

    def snapshot(self) -> Dict:
        return {"eta": self.eta, "w": self.w, "b": self.b,
                "err": [self._err_mean, self._err_m2, self._n]}

    def restore(self, state: Dict) -> None:
        self.eta = state["eta"]; self.w = state["w"]; self.b = state["b"]
        self._err_mean, self._err_m2, self._n = state["err"]


class PredictionLog:
    """Per-tick §30 records for the three targets, streamed to JSONL."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._fh = open(path, "w") if path else None
        self.cal_next = CalibrationTracker()
        self.cal_reward = CalibrationTracker()
        self.cal_useful = CalibrationTracker()
        self.n_ticks = 0

    def _emit(self, rec: Dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def record_next_obs(self, tick: int, episode: int, prediction: List[float],
                        confidence: float, uncertainty: float,
                        actual: List[float], update_norm: float) -> Dict:
        signed = [a - p for a, p in zip(actual, prediction)]
        abs_err = math.sqrt(sum(s * s for s in signed))
        mean_abs = sum(abs(s) for s in signed) / max(1, len(signed))
        self.cal_next.observe(confidence, mean_abs, uncertainty)
        rec = {"tick": tick, "episode": episode, "target": "next_obs",
               "prediction": prediction, "confidence": confidence,
               "uncertainty": uncertainty, "actual": actual,
               "signed_error": signed, "abs_error": abs_err,
               "mean_abs_error": mean_abs, "update_norm": update_norm}
        self._emit(rec)
        self.n_ticks += 1
        return rec

    def record_action_consequence(self, tick: int, episode: int,
                                  prediction: float, confidence: float,
                                  uncertainty: float, actual: float,
                                  update_norm: float) -> Dict:
        signed = actual - prediction
        self.cal_reward.observe(confidence, abs(signed), uncertainty)
        rec = {"tick": tick, "episode": episode,
               "target": "action_consequence",
               "prediction": prediction, "confidence": confidence,
               "uncertainty": uncertainty, "actual": actual,
               "signed_error": signed, "abs_error": abs(signed),
               "update_norm": update_norm}
        self._emit(rec)
        return rec

    def record_retrieval_usefulness(self, tick: int, episode: int,
                                    prediction: float, confidence: float,
                                    uncertainty: float, actual: float,
                                    update_norm: float) -> Dict:
        signed = actual - prediction
        self.cal_useful.observe(confidence, abs(signed), uncertainty)
        rec = {"tick": tick, "episode": episode,
               "target": "retrieval_usefulness",
               "prediction": prediction, "confidence": confidence,
               "uncertainty": uncertainty, "actual": actual,
               "signed_error": signed, "abs_error": abs(signed),
               "update_norm": update_norm}
        self._emit(rec)
        return rec

    def calibration_summary(self) -> Dict:
        return {
            "next_obs": self.cal_next.summary(),
            "action_consequence": self.cal_reward.summary(),
            "retrieval_usefulness": self.cal_useful.summary(),
            "n_ticks": self.n_ticks,
        }

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
