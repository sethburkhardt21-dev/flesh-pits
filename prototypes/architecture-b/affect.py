"""Affect — Architecture B (§33).

Two competing controllers drive the SAME modulation knobs:
  explore_gain : multiplier on the active-inference IG weight (curiosity)
  learn_gain   : multiplier on learning rates (plasticity under surprise)
  persist      : exploitation bias added to predicted reward (stay/switch)

1. ErrorAffect (the Architecture B claim): valence/arousal DERIVED from
   error dynamics — NOT a PAD dictionary lookup.
     valence = tanh(k * (slow_ema(|e|) - fast_ema(|e|)))
               positive when error is DECLINING, negative when rising
     arousal = clip(fast_ema(|d|e|/dt|) + mean_uncertainty, 0, 1)
   explore_gain = 1 + arousal
   learn_gain   = 1 + max(0, -valence)      (rising error -> learn harder)
   persist      = max(0, valence) * 0.5     (falling error -> keep course)

2. PadController (the §33 baseline to beat): donor-exact PAD dictionary
   mapping (EMOTION_PAD_MAP) with EMA dynamics driven by reward, novelty
   (= surprise = |e0|), and stress (= uncertainty). Same knobs:
     explore_gain = 1 + arousal
     learn_gain   = 1 + max(0, -valence)
     persist      = max(0, valence) * 0.5

K4 runs both as competing controllers on a foraging/avoidance task
(resource_world). If PAD wins or ties, the error-dynamic affect story is
dropped and PAD kept as the cheaper baseline.

No identity strings, no privileged tokens. Deterministic. Stdlib only.
"""

from __future__ import annotations

import math
from typing import Dict

# Donor-exact PAD label coordinates (reference: pad_emotion_core.py,
# itself a clean-room port of the_consciousness_ai's emotional_processing).
EMOTION_PAD_MAP: Dict[str, Dict[str, float]] = {
    "joy":      {"valence": 0.8,  "arousal": 0.5,  "dominance": 0.6},
    "sadness":  {"valence": -0.7, "arousal": -0.3, "dominance": -0.5},
    "anger":    {"valence": -0.6, "arousal": 0.8,  "dominance": 0.7},
    "fear":     {"valence": -0.7, "arousal": 0.7,  "dominance": -0.6},
    "surprise": {"valence": 0.2,  "arousal": 0.8,  "dominance": 0.0},
    "disgust":  {"valence": -0.6, "arousal": 0.3,  "dominance": 0.4},
    "trust":    {"valence": 0.6,  "arousal": -0.2, "dominance": 0.3},
    "neutral":  {"valence": 0.0,  "arousal": 0.0,  "dominance": 0.0},
}


def _tanh(x: float) -> float:
    e = math.exp(-2.0 * abs(x))
    s = (1.0 - e) / (1.0 + e)
    return s if x >= 0 else -s


def _knobs(valence: float, arousal: float) -> Dict[str, float]:
    return {
        "valence": valence,
        "arousal": arousal,
        "explore_gain": 1.0 + max(0.0, arousal),
        "learn_gain": 1.0 + max(0.0, -valence),
        "persist": max(0.0, valence) * 0.5,
    }


class ErrorAffect:
    """Valence/arousal derived from prediction-error dynamics."""

    def __init__(self, fast_alpha: float = 0.3, slow_alpha: float = 0.03,
                 k: float = 8.0) -> None:
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.k = k
        self._fast = 0.0
        self._slow = 0.0
        self._d_ema = 0.0
        self._prev_e = 0.0
        self._unc_ema = 0.0
        self.valence = 0.0
        self.arousal = 0.0

    def update(self, e_norm: float, uncertainty: float) -> Dict[str, float]:
        """One tick from the scalar error magnitude and model uncertainty."""
        e = float(e_norm)
        d = abs(e - self._prev_e)
        self._prev_e = e
        self._fast += self.fast_alpha * (e - self._fast)
        self._slow += self.slow_alpha * (e - self._slow)
        self._d_ema += self.fast_alpha * (d - self._d_ema)
        self._unc_ema += self.slow_alpha * (uncertainty - self._unc_ema)
        self.valence = _tanh(self.k * (self._slow - self._fast))
        self.arousal = max(0.0, min(1.0, self._d_ema * 4.0 + self._unc_ema))
        return _knobs(self.valence, self.arousal)

    def snapshot(self) -> Dict:
        return {"kind": "error", "fast": self._fast, "slow": self._slow,
                "d_ema": self._d_ema, "prev_e": self._prev_e,
                "unc_ema": self._unc_ema,
                "valence": self.valence, "arousal": self.arousal}

    def restore(self, state: Dict) -> None:
        if state.get("kind") != "error":
            raise ValueError("affect kind mismatch on restore")
        self._fast = state["fast"]; self._slow = state["slow"]
        self._d_ema = state["d_ema"]; self._prev_e = state["prev_e"]
        self._unc_ema = state["unc_ema"]
        self.valence = state["valence"]; self.arousal = state["arousal"]


class PadController:
    """PAD-dictionary baseline controller (donor-exact mapping + EMA)."""

    def __init__(self, alpha: float = 0.3,
                 reward_sensitivity: float = 0.4) -> None:
        self.alpha = alpha
        self.reward_sensitivity = reward_sensitivity
        self.pad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

    def _ema(self, cur: float, tgt: float) -> float:
        return cur + self.alpha * (tgt - cur)

    def update(self, reward: float, novelty: float,
               stress: float) -> Dict[str, float]:
        """Donor-exact update: reward->valence, novelty->arousal,
        stress->arousal up / dominance down, then decay toward neutral."""
        v, a, d = self.pad["valence"], self.pad["arousal"], self.pad["dominance"]
        v = self._ema(v, max(-1.0, min(1.0, reward * self.reward_sensitivity)))
        a = self._ema(a, max(0.0, min(1.0, novelty)))
        if stress > 0:
            a = self._ema(a, max(0.0, min(1.0, stress)))
            d = self._ema(d, max(-1.0, min(1.0, -stress * 0.5)))
        decay = 0.98
        self.pad = {"valence": v * decay, "arousal": a * decay,
                    "dominance": d * decay}
        kn = _knobs(self.pad["valence"], self.pad["arousal"])
        kn["dominance"] = self.pad["dominance"]
        return kn

    def snapshot(self) -> Dict:
        return {"kind": "pad", "pad": dict(self.pad)}

    def restore(self, state: Dict) -> None:
        if state.get("kind") != "pad":
            raise ValueError("affect kind mismatch on restore")
        self.pad = dict(state["pad"])
