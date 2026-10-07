"""K12 cue-structure battery env — parameterized changing_rule.

NAME: k12_cue_battery | VERSION: 1.0.0

Subclasses the canonical ChangingRule (experiments/envs/changing_rule.py)
and adds battery knobs; canonical knob setting
  n_cues=2, contingency="interaction", timing="pertick",
  cue_noise=0.0, flips=True
is EXACTLY changing_rule v1.0.0 (verified byte-equivalent in
experiments/k12_cue_structure_battery.py's preflight check).

Knobs:
  n_cues       : cardinality of the cue alphabet (2 / 4 / 8)
  contingency  : "interaction" (a* = rule[hidden], canonical),
                 "magnitude" (a* = fixed_action always; correct reward =
                   MAGNITUDE[observed_cue] — cue gates magnitude only),
                 "mag_interaction" (a* = rule[hidden] AND correct reward =
                   MAGNITUDE[observed_cue] — interaction with magnitude),
                 "distractor" (a* = fixed_action; correct reward = 1.0;
                   the cue is iid and causally irrelevant)
  timing       : "pertick" (iid cue per step, canonical),
                 "episodic" (cue drawn once per episode, constant),
                 "delayed" (obs["cue"] = None for steps 0..19, iid cue
                   for steps 20..39),
                 "t0only" (obs["cue"] visible at step 0 only, None after)
  cue_noise    : P(observed cue != hidden conditioning variable) per
                 observation (0.0 deterministic; 0.5 uninformative)
  flips        : rule flips every PHASE_LEN episodes (canonical True)

Observation "cue" space is categorical over range(n_cues) + [None];
None = cue hidden on this step (timing conditions). The context_fn in
the battery maps obs["cue"] -> obs.get("cue") so hidden steps land in
the None context, exactly like the K10 no-cue control.

The hidden conditioning variable `h` is drawn per step (pertick /
delayed) or per episode (episodic / t0only); the observed cue is a
possibly-noisy copy of h. Rewards are computed from h (interaction) or
from the observed cue (magnitude) as documented above; info carries
ground truth for ANALYSIS ONLY.

Stdlib only. Deterministic given (seed, episode_idx, action sequence).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_interface import new_rng  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402


class K12CueBattery(ChangingRule):
    NAME = "k12_cue_battery"
    VERSION = "1.0.0"

    MAGNITUDE_LO = 0.6
    MAGNITUDE_HI = 1.4  # magnitude schedule {cue0: LO, cue1: HI}
    FIXED_ACTION = 0    # a* for magnitude/distractor contingencies

    def __init__(self, *, n_cues=2, contingency="interaction",
                 timing="pertick", cue_noise=0.0, flips=True):
        if contingency not in ("interaction", "magnitude", "mag_interaction",
                               "distractor"):
            raise ValueError(contingency)
        if timing not in ("pertick", "episodic", "delayed", "t0only"):
            raise ValueError(timing)
        if not (0.0 <= cue_noise <= 0.5):
            raise ValueError(cue_noise)
        super().__init__()
        self._bn = int(n_cues)
        self._contingency = contingency
        self._timing = timing
        self._cue_noise = float(cue_noise)
        self._flips = bool(flips)
        self._h = None          # hidden conditioning variable
        self._cue = None        # observed cue (None = hidden)

    def observation_space(self):
        space = super().observation_space()
        vals = list(range(self._bn)) + [None]
        space["cue"] = {
            "type": "categorical", "values": vals,
            "desc": "battery cue, None = hidden on this step (timing)"}
        if self._contingency in ("magnitude", "mag_interaction"):
            # correct rewards reach MAGNITUDE_HI
            space["last_reward"] = {
                "type": "scalar", "low": 0.0, "high": self.MAGNITUDE_HI,
                "desc": "reward of previous step, 0 at t=0"}
        return space

    # -- lifecycle ----------------------------------------------------
    def reset(self, seed: int) -> dict:
        self._rng = new_rng(seed)
        self._seed = seed
        self._steps = 0
        self._episode_idx += 1
        phase = self._episode_idx // self.PHASE_LEN if self._flips else 0
        # Canonical rule schedule, generalized to n_cues (identical to
        # ChangingRule.reset when n_cues=2 and flips=True).
        rule_rng = new_rng((phase * 7919 + 0x5EED) % (2 ** 31))
        self._rule = [rule_rng.randrange(2) for _ in range(self._bn)]
        self._flipped = (self._last_phase is not None
                         and phase != self._last_phase)
        self._last_phase = phase
        self._phase = phase
        self._last_reward = 0.0
        self._last_action = 2
        if self._timing == "episodic":
            self._episode_cue = self._rng.randrange(self._bn)  # per-episode
        else:
            self._episode_cue = None
        self._h = None  # force re-draw: stale previous-episode h must not
        self._draw_h_and_cue()  # persist (episodic/t0only) from here on
        return self._obs()

    def _draw_h_and_cue(self):
        """Draw hidden h and the observed (possibly noisy) cue."""
        if self._timing in ("episodic", "t0only") and self._steps > 0:
            # h persists for the episode; the OBSERVED cue is still drawn
            # through the noise channel so timing only affects visibility.
            h = self._h if self._h is not None else self._episode_cue
        else:
            h = self._episode_cue if self._timing == "episodic" \
                else self._rng.randrange(self._bn)
        self._h = h
        self._cue = self._noisy_copy(h)

    def _noisy_copy(self, h):
        if self._cue_noise <= 0.0 or self._rng.random() >= self._cue_noise:
            return h
        if self._bn == 2:
            return 1 - h
        # general: resample a different value
        return (h + 1 + self._rng.randrange(self._bn - 1)) % self._bn

    def _cue_visible(self):
        if self._timing == "delayed":
            return self._steps >= 20
        if self._timing == "t0only":
            return self._steps == 0
        return True

    def step(self, action: int):
        action = self.validate_action(action)
        # --- contingency ------------------------------------------------
        if self._contingency in ("interaction", "mag_interaction"):
            correct = self._rule[self._h]
            base = 1.0
        elif self._contingency == "magnitude":
            correct = self.FIXED_ACTION
            base = (self.MAGNITUDE_LO if self._cue == 0
                    else self.MAGNITUDE_HI)
        else:  # distractor
            correct = self.FIXED_ACTION
            base = 1.0
        if self._contingency == "mag_interaction":
            base = (self.MAGNITUDE_LO if self._cue == 0
                    else self.MAGNITUDE_HI)
        reward = base if action == correct else 0.0
        if self._rng.random() < self.NOISE:
            # canonical 10% noise flip (around the realized reward)
            reward = ((base - reward)
                      if self._contingency in ("magnitude", "mag_interaction")
                      else (1.0 - reward))
        # --- advance ----------------------------------------------------
        self._last_reward = reward
        self._last_action = action
        self._steps += 1
        self._draw_h_and_cue()

        done = self._steps >= self.STEPS
        info = {"truncated": True} if done else {}
        # Ground truth for ANALYSIS ONLY — contract forbids agents using info.
        info.update({"phase": self._phase, "rule": list(self._rule),
                     "flipped_this_episode": self._flipped,
                     "episode_idx": self._episode_idx,
                     "hidden_h": self._h,
                     "observed_cue": self._cue,
                     "cue_visible": self._cue_visible()})
        return self._obs(), reward, done, info

    def _obs(self):
        obs = {"cue": (self._cue if self._cue_visible() else None),
               "last_reward": self._last_reward,
               "last_action": self._last_action}
        return self.validate_obs(obs, self.observation_space())

    def _canonical_state(self):
        return {
            "name": self.NAME, "version": self.VERSION, "seed": self._seed,
            "episode_idx": self._episode_idx, "phase": self._phase,
            "rule": self._rule, "hidden_h": self._h,
            "observed_cue": self._cue, "cue_visible": self._cue_visible(),
            "last_reward": self._last_reward,
            "last_action": self._last_action, "steps": self._steps,
            "knobs": {"n_cues": self._bn,
                      "contingency": self._contingency,
                      "timing": self._timing, "cue_noise": self._cue_noise,
                      "flips": self._flips},
            "rng": [self._rng.getstate()[0],
                    list(self._rng.getstate()[1]),
                    self._rng.getstate()[2]],
        }
