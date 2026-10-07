"""K11 — CUE-INDEXED ELIGIBILITY-TRACE COMBINATION (K9 x K10).

PREREGISTRATION (written 2026-10-07T03:59 EDT, BEFORE implementation/execution;
canonical copy: receipts/prereg_k11_cue_trace_combination.json):

  Hypothesis (H1): combining per-cue gain vectors (K10) with baseline-free
    eligibility-trace updates (K9) beats each parent mechanism alone on
    cue-conditioned delayed_reward (the intersection of NR-A-006 and
    NR-A-007), without degrading either parent task.
  Null (H0): the combination fails >= 1 gate on >= 2/4 seeds -> NR-A-012,
    with the predicted failure mechanism: within each cue's trace vector
    e_forward >> e_branch; forward-per-cue saturates to the 2.0 cap on
    shaping ticks alone; forward wins the t=0 arbitration (a no-op there);
    episodes truncate. Cue-indexing would fix WHICH vector fails, not
    WHETHER a state-blind bandit learns branch-then-forward sequencing.

Probes (fresh seeds {1212, 3434, 5656, 7878}, identical specialists /
theta / gain_lr everywhere):
  P1_main (cue_delayed_reward v1.0.0; 8 eps x <=15 steps; channels
    branch_a/branch_b/forward/stay; capacity=4):
      combined  = CueTraceArbitrator, cue context, baseline-free trace rule
      K10-arm   = CueIndexedArbitrator, cue context, original delta rule
      K9-arm    = SparseRewardArbitrator, NO context, trace rule
      frozen    = CueTraceArbitrator, frozen=True (pinned 1.0)
    Gates: G1 R_comb/K10 >= 1.15 AND G2 R_comb/frozen >= 1.30, >= 3/4.
    (G1 is the weak gate: the K10 arm is predicted to decay under sparse
    reward, possibly below frozen. G2 is binding.)
  P2_parent (delayed_reward v1.0.0; K5 P1 protocol):
      combined with CONSTANT context vs K9-arm (SparseRewardArbitrator).
    Non-degradation iff R = total(combined)/total(K9-arm) >= 0.90 on >=3/4.
  P3_parent (changing_rule v1.0.0; K10 B1 protocol; a0/a1; capacity=2):
      combined (cue ctx, trace rule) vs K10-arm (cue ctx, delta rule) +
      frozen control (combined class, frozen).
    Non-degradation iff R_comb/frozen >= 1.30 on >= 3/4. combined/K10
    head-to-head recorded as diagnostic.

Kill: any gate fails on majority of seeds -> the corresponding claim
dies -> record NR (main: NR-A-012 with measured mechanism).
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FLESH = os.path.dirname(os.path.dirname(TREE))  # flesh-pits/
sys.path.insert(0, os.path.join(FLESH, "experiments", "envs"))
sys.path.insert(0, os.path.join(FLESH, "experiments"))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from attention_cue import CueIndexedArbitrator
from attention_sparse import SparseRewardArbitrator
from attention_cue_trace import CueTraceArbitrator
from cue_delayed_reward import CueDelayedReward
from delayed_reward import DelayedReward
from changing_rule import ChangingRule
from k5_multitask_generalization import (CanonicalAdapter,
                                         neutral_specialists)

SEEDS = (1212, 3434, 5656, 7878)
THETA = 0.45
MARGIN_G1 = 1.15
MARGIN_G2 = 1.30
MARGIN_ND = 0.90
NEED = 3
TRACE_LAMBDA = 0.9
GAIN_LR = 0.15
EXPERIMENT_ID = "K11-cue-trace-combination"
DR_CHANNELS = ["branch_a", "branch_b", "forward", "stay"]
CR_CHANNELS = ["a0", "a1"]


def run_delayed_protocol(seed, arb_cls, frozen, env_obj, with_cue,
                         n_episodes=8):
    """K5 P1 / K9 protocol: delayed corridor, neutral specialists."""
    channels = DR_CHANNELS
    kwargs = {}
    if arb_cls is CueTraceArbitrator:
        assert abs(TRACE_LAMBDA - 0.9) < 1e-12
    context_fn = (lambda obs: int(obs["cue"])) if with_cue else None
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=4, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=arb_cls, context_fn=context_fn)
    assert isinstance(wk.arbitrator, arb_cls)
    if arb_cls is CueTraceArbitrator:
        assert abs(wk.arbitrator.trace_lambda - TRACE_LAMBDA) < 1e-12
    env = CanonicalAdapter(env_obj, seed,
                           {c: i for i, c in enumerate(channels)},
                           n_episodes=n_episodes)
    max_steps = env_obj.MAX_STEPS
    for _ in range(n_episodes * max_steps):
        wk.step(env.observe(), env)
    gains = (wk.arbitrator.all_context_gains()
             if hasattr(wk.arbitrator, "all_context_gains")
             else dict(wk.arbitrator.gains))
    return sum(wk.rewards), gains, len(wk.arbitrator.gain_history)


def run_changing_protocol(seed, arb_cls, frozen):
    """K10 B1 protocol: cue-conditioned changing_rule, cue in input."""
    channels = CR_CHANNELS
    context_fn = lambda obs: int(obs["cue"])  # noqa: E731
    wk = WorkspaceTick(channels, neutral_specialists(channels, seed),
                       capacity=2, frozen_gains=frozen, gain_lr=GAIN_LR,
                       ignition_kwargs={"theta": THETA},
                       arbitrator_cls=arb_cls, context_fn=context_fn)
    assert isinstance(wk.arbitrator, arb_cls)
    env = CanonicalAdapter(ChangingRule(), seed, {"a0": 0, "a1": 1},
                           n_episodes=12)
    for _ in range(12 * ChangingRule.STEPS):
        wk.step(env.observe(), env)
    gains = (wk.arbitrator.all_context_gains()
             if hasattr(wk.arbitrator, "all_context_gains")
             else dict(wk.arbitrator.gains))
    return sum(wk.rewards), gains, len(wk.arbitrator.gain_history)


def receipt_hash(rec):
    body = {k: v for k, v in rec.items() if k != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def ratio(a, b):
    return (a / b) if b > 0 else float("inf")


def main():
    t_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []

    def emit(kind, payload):
        rec = {"experiment_id": EXPERIMENT_ID, "kind": kind, "t_wall": t_wall}
        rec.update(payload)
        rec["prev_hash"] = (records[-1]["receipt_hash"]
                            if records else "GENESIS")
        rec["receipt_hash"] = receipt_hash(rec)
        records.append(rec)
        return rec

    def round_gains(g):
        if isinstance(g, dict) and g and isinstance(next(iter(g.values())),
                                                   dict):
            return {str(k): {c: round(v, 3) for c, v in vec.items()}
                    for k, vec in g.items()}
        return {k: round(v, 3) for k, v in g.items()}

    emit("preregistration", {
        "hypothesis": "H1: (K10 cue-indexed gains) x (K9 trace rule) beats "
                      "each parent alone on cue_delayed_reward AND does not "
                      "degrade either parent task",
        "null": "H0: combination fails >=1 gate on >=2/4 seeds -> NR-A-012 "
                "(predicted mechanism: per-cue forward saturation poisons "
                "the t=0 no-op, as in K9)",
        "gates": f"P1: G1 R_comb/K10 >= {MARGIN_G1} AND G2 R_comb/frozen "
                 f">= {MARGIN_G2} on >= {NEED}/{len(SEEDS)}; "
                 f"P2: R_comb/K9 >= {MARGIN_ND} on >= {NEED}/{len(SEEDS)}; "
                 f"P3: R_comb/frozen >= {MARGIN_G2} on >= {NEED}/{len(SEEDS)}",
        "seeds": list(SEEDS),
        "prereg_file": "receipts/prereg_k11_cue_trace_combination.json",
    })

    print(f"{EXPERIMENT_ID}")

    # ---------------- P1: main intersection ----------------
    print("--- P1: cue_delayed_reward (intersection)")
    g1_wins = g2_wins = 0
    for seed in SEEDS:
        t_comb, g_comb, n_comb = run_delayed_protocol(
            seed, CueTraceArbitrator, False, CueDelayedReward(), True)
        t_k10, g_k10, _ = run_delayed_protocol(
            seed, CueIndexedArbitrator, False, CueDelayedReward(), True)
        t_k9, g_k9, _ = run_delayed_protocol(
            seed, SparseRewardArbitrator, False, CueDelayedReward(), False)
        t_frz, _g_frz, _ = run_delayed_protocol(
            seed, CueTraceArbitrator, True, CueDelayedReward(), True)
        r_g1 = ratio(t_comb, t_k10)
        r_g2 = ratio(t_comb, t_frz)
        w1, w2 = r_g1 >= MARGIN_G1, r_g2 >= MARGIN_G2
        g1_wins += int(w1)
        g2_wins += int(w2)
        emit("seed_result", {
            "probe": "P1_cue_delayed_reward", "seed": seed,
            "combined_total": round(t_comb, 3),
            "k10_arm_total": round(t_k10, 3),
            "k9_arm_total": round(t_k9, 3),
            "frozen_total": round(t_frz, 3),
            "R_comb_over_k10": round(r_g1, 3), "G1_win": bool(w1),
            "R_comb_over_frozen": round(r_g2, 3), "G2_win": bool(w2),
            "combined_context_gains": round_gains(g_comb),
            "k10_context_gains": round_gains(g_k10),
            "k9_gains": round_gains(g_k9),
            "n_gain_updates_combined": n_comb,
        })
        print(f"  seed={seed}: comb={t_comb:.2f} k10={t_k10:.2f} "
              f"k9={t_k9:.2f} frz={t_frz:.2f} | "
              f"G1={r_g1:.2f} {'WIN' if w1 else 'loss'}  "
              f"G2={r_g2:.2f} {'WIN' if w2 else 'loss'}")

    # ---------------- P2: parent delayed_reward ----------------
    print("--- P2: delayed_reward (K9 parent, equivalence control)")
    p2_wins = 0
    for seed in SEEDS:
        t_comb, g_comb, _ = run_delayed_protocol(
            seed, CueTraceArbitrator, False, DelayedReward(), False)
        t_k9, g_k9, _ = run_delayed_protocol(
            seed, SparseRewardArbitrator, False, DelayedReward(), False)
        r = ratio(t_comb, t_k9)
        w = r >= MARGIN_ND
        p2_wins += int(w)
        emit("seed_result", {
            "probe": "P2_delayed_reward_parent", "seed": seed,
            "combined_constctx_total": round(t_comb, 3),
            "k9_arm_total": round(t_k9, 3),
            "R_comb_over_k9": round(r, 3), "nondegradation_win": bool(w),
            "combined_gains": round_gains(g_comb),
            "k9_gains": round_gains(g_k9),
        })
        print(f"  seed={seed}: comb(const)={t_comb:.2f} k9={t_k9:.2f} "
              f"R={r:.2f} {'OK' if w else 'DEGRADED'}")

    # ---------------- P3: parent changing_rule ----------------
    print("--- P3: changing_rule (K10 parent)")
    p3_wins = 0
    for seed in SEEDS:
        t_comb, g_comb, _ = run_changing_protocol(
            seed, CueTraceArbitrator, False)
        t_k10, g_k10, _ = run_changing_protocol(
            seed, CueIndexedArbitrator, False)
        t_frz, _g_frz, _ = run_changing_protocol(
            seed, CueTraceArbitrator, True)
        r_nd = ratio(t_comb, t_frz)
        r_h2h = ratio(t_comb, t_k10)
        w = r_nd >= MARGIN_G2
        p3_wins += int(w)
        emit("seed_result", {
            "probe": "P3_changing_rule_parent", "seed": seed,
            "combined_total": round(t_comb, 2),
            "k10_arm_total": round(t_k10, 2),
            "frozen_total": round(t_frz, 2),
            "R_comb_over_frozen": round(r_nd, 3),
            "nondegradation_win": bool(w),
            "R_comb_over_k10_diag": round(r_h2h, 3),
            "combined_context_gains": round_gains(g_comb),
            "k10_context_gains": round_gains(g_k10),
        })
        print(f"  seed={seed}: comb={t_comb:.1f} k10={t_k10:.1f} "
              f"frz={t_frz:.1f} | ND R={r_nd:.2f} "
              f"{'OK' if w else 'DEGRADED'} (h2h vs K10: {r_h2h:.2f})")

    # ---------------- verdict ----------------
    main_won = (g1_wins >= NEED) and (g2_wins >= NEED)
    p2_ok = p2_wins >= NEED
    p3_ok = p3_wins >= NEED
    if main_won and p2_ok and p3_ok:
        verdict, interp = ("COMBINATION WINS",
                           "G1 and G2 pass; both parent tasks non-degraded")
    elif main_won:
        bad = [p for p, ok in (("P2", p2_ok), ("P3", p3_ok)) if not ok]
        verdict, interp = ("PARTIAL: main win with parent-task trade-off",
                           f"main won but degraded: {bad}")
    else:
        verdict, interp = ("BOUND STANDS -> NR-A-012",
                           "combination fails >= 1 main gate; record NR "
                           "with measured mechanism")
    emit("verdict", {"G1_wins": g1_wins, "G2_wins": g2_wins,
                     "P2_wins": p2_wins, "P3_wins": p3_wins,
                     "need": NEED,
                     "main_won": bool(main_won),
                     "parent_nondegradation": bool(p2_ok and p3_ok),
                     "verdict": verdict, "interpretation": interp})
    print(f"  G1: {g1_wins}/{len(SEEDS)}  G2: {g2_wins}/{len(SEEDS)}  "
          f"P2: {p2_wins}/{len(SEEDS)}  P3: {p3_wins}/{len(SEEDS)}")
    print(f"  -> {verdict}")

    out = os.path.join(TREE, "receipts", "k11_cue_trace_combination.ndjson")
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    print("receipt:", out)


if __name__ == "__main__":
    main()
