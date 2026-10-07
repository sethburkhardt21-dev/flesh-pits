# Negative results — Architecture A (Phase 4 gap-closure, 2026-10-07)

Supplements the lane-level `flesh-pits/research/negative_results.md`.
NR-A-001/002/003 (build-time mechanism bugs), NR-A-004 (theta=0.6
stationarity perseveration, RESOLVED), NR-A-005 (frozen change-bids suffice
on signal-tracking reversals) and NR-A-010 (K8 gate-a1 harness artifact)
were cited from ARCHITECTURE_A.md / MATURITY.md but their full entries were
never written anywhere — reconstructed 2026-10-07 by the maturity audit and
appended below, marked as reconstructions. New entries follow the same
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

## 2026-10-07 — NR-A-001/002/003 (reconstructed): build-time mechanism bugs

*Reconstruction note (maturity audit, 2026-10-07): ARCHITECTURE_A.md cites
NR-A-001/002/003 as living in this file and this file's header defers them
to the lane level — neither location ever held full entries. The entries
below are reconstructed from the one-line descriptions in ARCHITECTURE_A.md
("tracking baselines, unobserved-arm punishment, within-tick ignition
contamination"); all three were fixed during the build, before K1–K4.*

- **NR-A-001 — tracking baselines**: a baseline that tracked the wrong
  statistic (all-history instead of recency-weighted) admitted almost
  nothing after the initial transient. Replaced with a recency-weighted
  baseline; surprise is relative to current expectations.
- **NR-A-002 — unobserved-arm punishment**: the gain update punished arms
  that were never observed, decaying their gains for lack of evidence
  rather than evidence of lack. Fixed: only observed arms update.
- **NR-A-003 — within-tick ignition contamination**: ignition state leaked
  across the tick boundary, letting one tick's gate decision contaminate
  the next tick's arbitration. Fixed: per-tick gate state isolation.

## 2026-10-07 — NR-A-004: theta=0.6 stationarity perseveration (RESOLVED, K6+K8)

- Expectation: with the ignition gate at theta=0.6, the agent would still
  act when the environment went fully stationary — ignition selects, the
  gate propagates, the loop continues.
- Observed: under sustained stationarity nothing crossed the gate, so no
  proposal ever reached action selection and the agent froze on the stale
  channel (K6 baseline: phase-2 'c' fraction 0.0, total ~84–85 while the
  relevant channel had reversed). The freeze is the price of the gate as
  built — action selection was gated on ignition.
- Resolution: R1 sub-ignition exploratory path — when nothing ignites, act
  on the graded arbitration winner WITHOUT propagating anything to
  consumers (gate still decides ALL propagation; K2 sole-path unaffected).
  K6 (seeds 1111/2222/3333): phase-2 'c' fraction 0.0→0.78, total 85→128.
  K8 (fresh seeds 51501–51503): all 5 preregistered gates PASS; R1 wired as
  the default `_select_action` 2026-10-07.
- Rules out: "the freeze is the price of the gate." R2 (stationarity
  detector lowering theta) also resolves but admits ~2.7× ignitions,
  weakening gate selectivity — rejected.
- Receipts: receipts/k6_stationarity_resolution.json,
  receipts/k8_r1_wiring_confirmation.json,
  receipts/k8_r1_wiring_decision.json.

## 2026-10-07 — NR-A-005 (reconstructed): frozen change-bids suffice on signal-tracking reversals

*Reconstruction note (maturity audit, 2026-10-07): cited as the bound on
K4's claim in ARCHITECTURE_A.md and MATURITY.md; no full entry was ever
written. Reconstructed from those citations.*

- Expectation: learned attention gains would beat frozen gains on
  signal-tracking reversal tasks.
- Observed: on signal-tracking reversals the frozen change-bid baseline
  adapts alone — no learned/frozen gap. The K4 win is specific to
  salience-orthogonal relevance shifts.
- Rules out: "learned attention earns its keep on every reversal task."
  Precise bound: learning earns its keep for salience-orthogonal relevance
  shifts (generalized K5 to noisy-signal tracking P2 and multi-reversal
  stationary shifts P3), not where the frozen bid dynamics already track.

## 2026-10-07 — NR-A-010: K8 first-run gate-a1 harness artifact (methodological)

- Expectation: the K8 gate (a1) — frozen-gains ignition trajectories
  EXACTLY identical baseline vs R1 on 3/3 seeds — would pass on the first
  implementation.
- Observed: FAILED on seed 51502 (ws-586 vs ws-1186) despite identical
  ignition decisions. Root cause: `workspace_buffer.BoundedWorkspace._ids`
  is a process-global `itertools.count`, so the second condition in the
  same process continued the first condition's numbering. Harness
  artifact, not a behavioral difference — verified by hand: same tick,
  same bids, same winner, one ignition each.
- Correction: reset the counter between conditions and compare behavioral
  content per tick (ignited item id+strength, competed bids, winner). The
  preregistration is unchanged; re-run passed 5/5 gates → WIRE.
- Lesson: identity comparisons must name WHAT is being compared
  (behavioral content, not process-global sequence numbers).
- Follow-up note: a post-wire re-run of K8 is degenerate — the default
  tick now includes the R1 path, so the baseline-vs-candidate contrast
  collapses (verified by the auditor's own hand: gates a1–d pass, gate e
  fails trivially because baseline now recovers too). The WIRE decision
  rests on the pre-wire run recorded in the decision receipt.
- Receipt: receipts/k8_r1_wiring_decision.json (method_correction field).
