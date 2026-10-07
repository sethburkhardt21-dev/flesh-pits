"""K4 — FIXED-ATTENTION BASELINE (Architecture A load-bearing claim 4).

Hypothesis: learned attention (gains adapted from utility history via
  the broadcast feedback loop) beats fixed salience rules on a
  changing-relevance task.
Null: no performance gap (attention learning is decorative; keep the
  cheaper fixed rules).
Preregistered metric: total reward ratio R = total(learned) /
  total(fixed) over 200 ticks. PASS requires R >= 1.30 on at least 3
  of 4 seeds.
Task (stationary-signal changing-relevance): 3 channels; observations
  are stationary (0.5 + noise on every channel — salience carries NO
  relevance information); latent reward weights concentrate on channel
  'a' for ticks 1-100 then REVERSE to channel 'c' for ticks 101-200.
  The change-bid mechanism alone cannot track this (no signal change);
  only reward-driven gain adaptation can.
Baseline: identical tick with arbitrator frozen (gains pinned at 1.0;
  winner = argmax raw habituated bid ~ noise + tie-break).
  NOTE: this baseline is honest but not straw: the z-score bids are
  themselves adaptive; "fixed" refers to the learned gains only.
Ablation: frozen gains ARE the ablation of the learning loop.
Procedure: 200 ticks/condition, seeds {11, 22, 33, 44}, theta=0.45
  (at theta=0.6 the architecture perseverates under stationarity —
  documented limitation, not part of this comparison; both conditions
  run at the same theta).
Seed: listed per run.
Kill: R < 1.30 on majority of seeds -> attention learning is
  decorative -> keep fixed rules, remove the learning machinery.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists

SEEDS = (11, 22, 33, 44)
TICKS = 200
THETA = 0.45
MARGIN = 1.30
NEED = 3  # of 4 seeds


def run(frozen, seed):
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       frozen_gains=frozen, gain_lr=0.15,
                       ignition_kwargs={"theta": THETA})
    env = ChangingRelevanceEnv(channels, TICKS, seed=seed,
                               stationary_signals=True)
    for _ in range(TICKS):
        wk.step(env.observe(), env)
    total = sum(wk.rewards)
    p2 = sum(wk.rewards[TICKS // 2:])
    return total, p2, wk.actions, dict(wk.arbitrator.gains)


def main():
    print("K4 — learned attention vs fixed salience rules "
          "(changing-relevance, stationary signals)")
    print(f"preregistered: R=total(learned)/total(fixed) >= {MARGIN} "
          f"on >= {NEED}/{len(SEEDS)} seeds")
    results, wins = {}, 0
    for seed in SEEDS:
        tl, p2l, al, gl = run(False, seed)
        tf, p2f, af, gf = run(True, seed)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= MARGIN
        wins += int(win)
        results[str(seed)] = {"learned_total": round(tl, 2),
                              "fixed_total": round(tf, 2),
                              "ratio": round(r, 3), "win": win,
                              "learned_p2": round(p2l, 2),
                              "fixed_p2": round(p2f, 2),
                              "learned_final_actions": al[-10:],
                              "learned_gains": {k: round(v, 3)
                                                for k, v in gl.items()}}
        print(f"  seed={seed}: learned={tl:.1f} (p2={p2l:.1f}) "
              f"fixed={tf:.1f} (p2={p2f:.1f}) R={r:.2f} "
              f"{'WIN' if win else 'loss'}")
    gate = wins >= NEED
    verdict = ("PASS — learned attention beats fixed salience rules "
               f"({wins}/{len(SEEDS)} seeds at R>={MARGIN}); the learning "
               "loop is load-bearing" if gate else
               "FAIL — kill condition met: no performance gap; attention "
               "learning is decorative — keep fixed rules")
    print(f"wins: {wins}/{len(SEEDS)} (need {NEED})")
    print("VERDICT:", verdict)
    receipt = {"experiment": "K4_attention_baseline", "seeds": list(SEEDS),
               "ticks": TICKS, "theta": THETA, "margin": MARGIN,
               "need": NEED, "results": results, "wins": wins,
               "pass": bool(gate), "verdict": verdict,
               "limitation": ("at theta=0.6 the architecture perseverates "
                              "under stationary signals (no ignition -> no "
                              "planner proposals -> action holds); both "
                              "conditions ran at theta=0.45")}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "receipts", "k4_attention_baseline.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
