"""K5 — MULTI-TASK GENERALIZATION OF LEARNED ATTENTION (Phase 4 gap 1).

K4's claim is bounded: the learned-gain win is proven only on the
stationary-signal changing-relevance task family. This experiment tests
the learned-gain loop on three further task families.

Hypothesis: learned gains (utility-driven, via broadcast feedback) beat
  frozen gains on task families beyond K4's.
Null: the learned loop's win is specific to the K4 task family; on other
  families the frozen baseline matches (the bound generalizes like
  NR-A-005 did).
Preregistered metric (per probe): R = total(learned) / total(frozen)
  over the probe's ticks. Per-probe PASS (claim generalizes to that
  family): R >= 1.30 on >= 3 of 4 fresh seeds {101, 202, 303, 404}.
  Where it fails, the bound is recorded honestly (NR-A-005 pattern).
Baseline: identical tick with arbitrator frozen (gains pinned at 1.0),
  same theta=0.45 (K4's non-freezing theta; the NR-A-004 freeze is
  E2's subject, not confounded here).
Ablation: frozen gains ARE the ablation of the learning loop.
Procedure: four probes:
  P1 delayed_reward (canonical flesh-pits env, adapted): 8 episodes x
      <=15 steps. Channels = action labels; neutral specialists
      (0.5 + noise) so selection is gain-driven (a bandit isolation of
      the gain loop). Tests temporal credit assignment.
  P2 noisy-signal: local ChangingRelevanceEnv, stationary_signals=False,
      noise=0.25 (heavy observation noise). Tests whether learning adds
      value when salience is noisy (NR-A-005 showed frozen suffices at
      noise=0.05).
  P3 multi-reversal: local 4-phase x 75-tick changing-relevance,
      stationary signals (salience carries no info — isolates the gain
      loop), weights alternate each phase. Tests repeated tracking
      (gain_cap=2.0 saturation risk).
  P4 changing_rule (canonical, adapted): 12 episodes x 40 steps,
      rule flips every 5 episodes. Neutral specialists; tests
      multi-reversal contingency tracking WITHOUT cue input (the
      architecture cannot condition on cue — expected bound).
Seed: 101, 202, 303, 404 (fresh; K1-K4 used 11/22/33/44).
Kill (per probe): R < 1.30 on majority of seeds -> the learned loop's
  win does not generalize to that family -> record the bound.
Interpretation/limitations: filled after the run.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
# TREE last = highest priority: local envs.py must shadow flesh-pits/envs/.
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists
from delayed_reward import DelayedReward
from changing_rule import ChangingRule

SEEDS = (101, 202, 303, 404)
THETA = 0.45
MARGIN = 1.30
NEED = 3


# ---------------------------------------------------------------- adapters
class CanonicalAdapter:
    """Adapts a canonical flesh-pits Environment to the Architecture A
    tick protocol: observe() -> dict for specialists, step(label) ->
    float reward. Episodes auto-reset on done."""

    def __init__(self, env, seed, channel_to_action, n_episodes):
        self.env = env
        self.channel_to_action = dict(channel_to_action)
        self.episode_seeds = [seed * 1000 + e for e in range(n_episodes)]
        self.ep = 0
        self.obs = self.env.reset(self.episode_seeds[0])
        self.episodes_done = 0

    def observe(self):
        return dict(self.obs)

    def step(self, action_label):
        a = self.channel_to_action[action_label]
        obs, reward, done, _info = self.env.step(a)
        self.obs = obs
        if done:
            self.episodes_done += 1
            self.ep += 1
            self.obs = self.env.reset(
                self.episode_seeds[self.ep % len(self.episode_seeds)])
        return float(reward)


def neutral_specialists(channels, seed, noise=0.05):
    """Neutral specialists: stimulus = 0.5 + noise on every channel.
    No task information in salience; selection is purely gain-driven,
    isolating the learned-gain loop (bandit framing). Deterministic."""
    specs = []
    for i, c in enumerate(channels):
        rng = random.Random(seed * 7919 + i * 131)
        n = noise

        def fn(obs, tick, _rng=rng, _n=n, _c=c):
            return max(0.0, min(1.0, 0.5 + _rng.gauss(0, _n))), {
                "summary": f"neutral:{_c}"}
        specs.append((c, fn))
    return specs


class MultiReversalEnv(ChangingRelevanceEnv):
    """4 phases x `phase_len` ticks; weights alternate each phase;
    stationary signals (salience carries no relevance info)."""

    def __init__(self, channels, phase_len, n_phases, *, seed=0, noise=0.05):
        self.n_phases = int(n_phases)
        self.phase_len = int(phase_len)
        super().__init__(channels, phase_len * n_phases, seed=seed,
                         noise=noise, stationary_signals=True)

    def weights(self):
        phase = min(self.tick // self.phase_len, self.n_phases - 1)
        w1, w2 = self.weights_phase
        return dict(zip(self.channels, w1 if phase % 2 == 0 else w2))


# ---------------------------------------------------------------- probes
def probe_delayed_reward(seed, frozen):
    channels = ["branch_a", "branch_b", "forward", "stay"]
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    env = CanonicalAdapter(DelayedReward(), seed,
                           {c: i for i, c in enumerate(channels)},
                           n_episodes=8)
    for _ in range(8 * DelayedReward.MAX_STEPS):
        wk.step(env.observe(), env)
    return sum(wk.rewards), dict(wk.arbitrator.gains)


def probe_noisy_signal(seed, frozen):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    env = ChangingRelevanceEnv(channels, 200, seed=seed, noise=0.25,
                               stationary_signals=False)
    for _ in range(200):
        wk.step(env.observe(), env)
    return sum(wk.rewards), dict(wk.arbitrator.gains)


def probe_multi_reversal(seed, frozen):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    env = MultiReversalEnv(channels, 75, 4, seed=seed)
    per_phase = []
    for p in range(4):
        r0 = sum(wk.rewards)
        for _ in range(75):
            wk.step(env.observe(), env)
        per_phase.append(round(sum(wk.rewards) - r0, 2))
    return sum(wk.rewards), per_phase, dict(wk.arbitrator.gains)


def probe_changing_rule(seed, frozen):
    channels = ["a0", "a1"]
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=2, frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    env = CanonicalAdapter(ChangingRule(), seed, {"a0": 0, "a1": 1},
                           n_episodes=12)
    for _ in range(12 * ChangingRule.STEPS):
        wk.step(env.observe(), env)
    return sum(wk.rewards), dict(wk.arbitrator.gains)


PROBES = {
    "P1_delayed_reward": probe_delayed_reward,
    "P2_noisy_signal": probe_noisy_signal,
    "P3_multi_reversal": probe_multi_reversal,
    "P4_changing_rule": probe_changing_rule,
}


def main():
    print("K5 — multi-task generalization of learned attention")
    print(f"preregistered: per probe, R=total(learned)/total(frozen) >= "
          f"{MARGIN} on >= {NEED}/{len(SEEDS)} seeds")
    all_results, verdicts = {}, {}
    for pname, pfn in PROBES.items():
        print(f"--- {pname}")
        results, wins = {}, 0
        for seed in SEEDS:
            out_l = pfn(seed, False)
            out_f = pfn(seed, True)
            tl, tf = out_l[0], out_f[0]
            r = tl / tf if tf > 0 else float("inf")
            win = r >= MARGIN
            wins += int(win)
            rec = {"learned_total": round(tl, 2),
                   "frozen_total": round(tf, 2),
                   "ratio": round(r, 3), "win": bool(win),
                   "learned_gains": {k: round(v, 3)
                                     for k, v in out_l[-1].items()}}
            if len(out_l) == 3:  # multi-reversal carries per-phase totals
                rec["learned_per_phase"] = out_l[1]
            results[str(seed)] = rec
            print(f"  seed={seed}: learned={tl:.1f} frozen={tf:.1f} "
                  f"R={r:.2f} {'WIN' if win else 'loss'}")
        passed = wins >= NEED
        verdicts[pname] = bool(passed)
        all_results[pname] = {"results": results, "wins": wins,
                              "pass": bool(passed)}
        print(f"  -> {'GENERALIZES' if passed else 'BOUND RECORDED'} "
              f"({wins}/{len(SEEDS)})")
    receipt = {"experiment": "K5_multitask_generalization",
               "seeds": list(SEEDS), "theta": THETA, "margin": MARGIN,
               "need": NEED, "probes": all_results, "verdicts": verdicts}
    out = os.path.join(TREE, "receipts", "k5_multitask_generalization.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
