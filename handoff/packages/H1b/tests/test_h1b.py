"""H1b package test — cue-indexed gain adapter (K10 procedure).

Standalone proof from the packaged sources:
  B1 (cue in input):  context_fn = cue -> per-cue gain vectors; learned must
                      beat frozen with R >= 1.30 on >= 3 of 4 seeds.
  B2 (cue withheld):  context_fn = constant -> the cue, not the module, must
                      be causal: NO gap (R < 1.30 on >= 3/4 seeds).
Verdict BOUND LIFTED iff B1 passes AND B2 is clean; CONFOUND if B2 gaps;
BOUND STANDS if B1 fails.

Seeds {5555,6666,7777,8888}, 12 episodes x 40 steps, theta=0.45,
gain_lr=0.15 — the K10 preregistration verbatim.

Exit 0 on BOUND LIFTED, 1 otherwise. Receipt -> ./receipts/.
Stdlib only. Deterministic (seeded).
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "support"))

from tick import WorkspaceTick
from attention_cue import CueIndexedArbitrator
from changing_rule import ChangingRule

SEEDS = (5555, 6666, 7777, 8888)
THETA = 0.45
MARGIN = 1.30
NEED = 3
GAIN_LR = 0.15
N_EPISODES = 12


# --- test support: copied verbatim from
# flesh-pits/prototypes/architecture-a/experiments/k5_multitask_generalization.py
# (CanonicalAdapter, neutral_specialists) — stdlib only, deterministic.
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
    No task information in salience; selection is purely gain-driven."""
    specs = []
    for i, c in enumerate(channels):
        rng = random.Random(seed * 7919 + i * 131)
        n = noise

        def fn(obs, tick, _rng=rng, _n=n, _c=c):
            return max(0.0, min(1.0, 0.5 + _rng.gauss(0, _n))), {
                "summary": f"neutral:{_c}"}
        specs.append((c, fn))
    return specs
# --- end copied support ---


def probe(seed, frozen, with_cue):
    channels = ["a0", "a1"]
    context_fn = (lambda obs: int(obs["cue"])) if with_cue else (
        lambda obs: None)
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=2, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=CueIndexedArbitrator,
                       context_fn=context_fn)
    assert isinstance(wk.arbitrator, CueIndexedArbitrator)
    env = CanonicalAdapter(ChangingRule(), seed, {"a0": 0, "a1": 1},
                           n_episodes=N_EPISODES)
    for _ in range(N_EPISODES * ChangingRule.STEPS):
        wk.step(env.observe(), env)
    return sum(wk.rewards)


def run_probe(name, with_cue):
    wins, rows = 0, {}
    for seed in SEEDS:
        tl = probe(seed, False, with_cue)
        tf = probe(seed, True, with_cue)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        rows[str(seed)] = {"learned": round(tl, 2), "frozen": round(tf, 2),
                           "ratio": round(r, 3), "win": win}
        print(f"  [{name}] seed={seed}: learned={tl:.1f} frozen={tf:.1f} "
              f"R={r:.2f} {'WIN' if win else 'loss'}")
    return wins, rows


def main():
    print("B1 — cue in input (bound-lift probe)")
    w1, r1 = run_probe("B1", True)
    print("B2 — cue withheld (confound control)")
    w2, r2 = run_probe("B2", False)
    main_pass = w1 >= NEED
    ctrl_clean = (len(SEEDS) - w2) >= NEED
    if main_pass and ctrl_clean:
        verdict, ok = "BOUND LIFTED", True
    elif main_pass and not ctrl_clean:
        verdict, ok = "CONFOUND (control also gaps — do NOT claim)", False
    else:
        verdict, ok = "BOUND STANDS (NR-A-007)", False
    print(f"B1 wins {w1}/{len(SEEDS)}; B2 gaps {w2}/{len(SEEDS)} -> {verdict}")
    receipt = {"package": "H1b", "test": "test_h1b.py",
               "seeds": list(SEEDS), "n_episodes": N_EPISODES,
               "steps": ChangingRule.STEPS, "theta": THETA,
               "margin": MARGIN, "need": NEED,
               "B1": {"wins": w1, "rows": r1},
               "B2": {"wins": w2, "rows": r2},
               "main_pass": bool(main_pass), "control_clean": bool(ctrl_clean),
               "verdict": verdict, "pass": bool(ok)}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
