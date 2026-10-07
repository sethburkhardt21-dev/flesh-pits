# Negative results — Architecture A (Phase 4 gap-closure, 2026-10-07)

Supplements the lane-level `flesh-pits/research/negative_results.md`.
NR-A-001/002/003 (build-time mechanism bugs) and NR-A-004 (theta=0.6
stationarity perseveration) / NR-A-005 (frozen change-bids suffice on
signal-tracking reversals) are referenced from ARCHITECTURE_A.md; their
full entries live at the lane level. New entries below follow the same
honest-bounds discipline as NR-A-005.

## 2026-10-07 — NR-A-006: learned attention does not generalize to delayed_reward (K5 P1)
- Expectation: the learned-gain loop would beat frozen gains on the
  canonical delayed_reward task (preregistered R >= 1.30, 3/4 seeds).
- Observed: R = 0.08–0.17 on 4/4 fresh seeds {101,202,303,404} — the
  learned condition LOSES outright. Learned totals 0.04–0.12 vs frozen
  0.36–0.80; both near zero.
- Mechanism: the constant-baseline (0.5) delta rule punishes every arm
  under sparse reward (per-tick rewards 0–0.02 all sit below baseline),
  so all gains decay toward the 0.01 floor; the delayed +1.0 is credited
  to the corridor-end action (forward), never to the t=0 branch choice.
  The loop as built requires per-tick reward correlated with channel
  utility above a fixed baseline — a structural precondition, not a
  tuning issue.
- Rules out: "learned attention generalizes to sparse delayed-reward
  tasks." Receipt: receipts/k5_multitask_generalization.json (P1).

## 2026-10-07 — NR-A-007: learned attention does not generalize to cue-conditioned changing_rule (K5 P4)
- Expectation: learned gains would track the canonical changing_rule's
  multi-reversal contingencies (preregistered R >= 1.30, 3/4 seeds).
- Observed: R = 1.03–1.09 on 4/4 seeds — no gap. Both conditions earn
  ~227–259/480 (~50%, chance): without cue input the architecture is a
  context-free bandit and no gain rule can beat chance on a
  cue-conditioned task.
- Rules out: "the gain loop earns its keep where the relevant state is
  not in its input." The bound is architectural (no state-conditioned
  policy), not a learning failure. Receipt:
  receipts/k5_multitask_generalization.json (P4).

## 2026-10-07 — NR-A-008 (methodological): K6 freeze_rate preregistration was mis-specified
- Expectation: freeze_rate = P(action_t == action_{t-1}) < 0.90 would
  operationalize "the freeze is eliminated."
- Observed: a perfect reversal-tracker holds each action ~100 ticks and
  scores freeze_rate ≈ 0.99 — the metric cannot distinguish "frozen"
  from "tracking." Both candidates scored 0.95–0.99 while fully
  resolving the NR-A-004 harm (phase-2 'c' fraction 0.0 → 0.78–0.79,
  total 85 → 128).
- Correction: superseded by phase2_c_fraction > 0.5 (does the agent act
  on the newly-relevant channel after the reversal). The preregistered
  numbers are kept in the receipt; the lesson is recorded here: a
  metric that a perfect solver fails is measuring the wrong thing.
  Receipt: receipts/k6_stationarity_resolution.json.

## 2026-10-07 — NR-A-009: K7 behavioral proxy shortfall (bound on the CAUSAL promotion)
- Expectation: the bid lesion would cut total reward by >= 1.30x
  (preregistered T >= 1.30, 3/4 seeds).
- Observed: T = 1.14–1.25 on 4/4 seeds {777,888,999,1212} — the
  preregistered bar FAILED 0/4. Mechanism-level the lesion is decisive
  (winner entropy 1.03–1.38 → exactly 0.0 bits: arbitration collapses
  to the tie-break), but the reward proxy is buffered two ways: (1) the
  task's reward structure pays ~85 to a fixed 'a' policy regardless;
  (2) learned gains fully compensate for dead bids (lesioned+learned
  128–130 vs intact+learned 139–140) — the K4 win is carried by the gain
  loop, not the bids.
- Rules out: "bids are load-bearing for learned-attention performance."
  Precise characterization: bids are CAUSAL for moment-to-moment
  arbitration under frozen gains; the learned system barely needs them.
  Receipt: receipts/k7_bid_ablation.json (verdict_split).
