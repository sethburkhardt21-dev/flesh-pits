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

## 2026-10-07 — NR-A-011: trace-rule redesign does not lift the delayed_reward bound (K9)

- Expectation (H1, preregistered): replacing the constant-baseline (0.5)
  delta rule with a return-conditioned, baseline-free eligibility-trace
  gain update (attention_sparse.py: per-tick e_c <- 0.9*e_c, e_winner += 1;
  on return r != 0, gain[c] += 0.15*r*e_c; no change on r == 0) would let
  learned gains beat frozen gains on canonical delayed_reward (R >= 1.30
  on >= 3/4 fresh seeds {1111, 2222, 3333, 4444}; every other
  hyperparameter identical to K5 P1).
- Observed: R = 0.50 / 0.34 / 0.43 / 0.47 — 0/4 wins, BOUND STANDS.
  Learned totals 0.10–0.34 vs frozen 0.20–0.82; both at shaping level (no
  +1.0 events in either condition on most seeds).
- Mechanism (measured): the trace rule operated as designed — forward gain
  rose to 1.24–1.29 on shaping-heavy seeds; branch channels received only
  lambda^10-discounted credit (negligible). Two structural facts killed it:
  (1) the +1.0 is too rare under near-uniform play for trace credit to
  matter — learned totals stayed at shaping level; (2) the mild forward
  preference the rule DID learn poisons the t=0 decision (forward is a
  no-op at t=0; each forward-win there wastes one of 15 steps), so learned
  < frozen on every seed. The preregistered H0 predicted the
  forward-credit concentration correctly; the realized dynamics were
  milder (t=0 poisoning, not full lock-in).
- Rules out: "a return-conditioned baseline-free gain update lifts
  NR-A-006." A state-blind bandit cannot learn branch-then-forward
  sequencing — changing the gain-update rule changes HOW it fails, not
  whether. The NR-A-006 bound is STRUCTURAL (input/state, not the rule).
- Receipt: receipts/k9_sparse_reward_redesign.ndjson (hash-chained:
  preregistration + 4 seed_results + verdict). Prereg:
  receipts/prereg_k9_sparse_reward_redesign.json.
- Second-lane replication (2026-10-07): independent driver
  (experiments/repro_second_lane_k9.py, rewritten from spec; no import of
  the original script), preregistration sealed BEFORE the driver was
  written (receipts/prereg_repro_second_lane_k9.json, seal
  237fd706e356bb9c). Fresh seeds {91511, 91512, 91513, 91514}: R = 0.32 /
  0.58 / 0.24 / 0.67 — 0/4 wins. VERDICT: REPRODUCES (BOUND STANDS again).
  NR-A-011 holds under independent replication. Receipt:
  receipts/repro_second_lane_k9.ndjson (hash-chained: preregistration + 4
  seed_results + verdict, chain verified).

## 2026-10-07 — NR-A-007 conditionally LIFTED (K10): cue-indexed gains

- The bound AS STATED replicates: K10 control probe B2 (CueIndexedArbitrator
  with constant context — cue withheld; code path otherwise identical)
  shows R = 0.95–1.13, 0/4 gaps on fresh seeds {5555, 6666, 7777, 8888},
  matching NR-A-007's 1.03–1.09. Without the cue in the input the
  architecture is a context-free bandit and no gain rule beats chance.
- With the cue in the input (B1: context_fn = obs["cue"] -> one gain
  vector per cue value; ORIGINAL delta rule unchanged): R = 1.52 / 1.68 /
  1.64 / 1.58 — 4/4 WIN (gate >= 1.30). Learned totals 368–392/480 vs
  frozen 230–244/480 (chance).
- Mechanism evidence: per-cue gain vectors diverged as designed (seed 5555
  final: cue0 {a0: 1.925, a1: 0.875}, cue1 {a0: 0.95, a1: 2.0}) — the
  K4/P3 reversal-tracking mechanism operating per cue, re-learning across
  the 5-episode rule flips.
- Precise current claim: NR-A-007 holds IFF the relevant state is absent
  from the input. It is an architectural INPUT bound, not a learning
  failure. With state-conditioned gains the bound lifts; the control
  confirms the cue (not the new module) is the causal factor.
- New modules: prototypes/architecture-a/attention_cue.py
  (CueIndexedArbitrator; attention.py untouched), tick.py gained additive
  `arbitrator_cls` / `context_fn` params (defaults = proven behavior).
- Receipt: receipts/k10_cue_indexed_adapter.ndjson (hash-chained:
  preregistration + 8 seed_results + verdict). Prereg:
  receipts/prereg_k10_cue_indexed_adapter.json.
- Second-lane replication (2026-10-07): independent driver
  (experiments/repro_second_lane_k10.py, rewritten from spec; no import of
  the original script), preregistration sealed BEFORE the driver was
  written (receipts/prereg_repro_second_lane_k10.json, seal
  48453d1a5cb85100). Fresh seeds {91615, 91616, 91617, 91618}: B1 R =
  1.45 / 1.58 / 1.56 / 1.47 — 4/4 wins; B2 control R = 0.88 / 1.06 /
  1.02 / 1.07 — 0/4 gaps. VERDICT: REPRODUCES (both probes: B1 lift and
  B2 no-gap control). The NR-A-007 conditional lift holds under
  independent replication; control remains clean (no module confound).
  Receipt: receipts/repro_second_lane_k10.ndjson (hash-chained:
  preregistration + 8 seed_results + verdict, chain verified).

## 2026-10-07 — NR-A-012: (trace-rule x cue-adapter) combination fails on the intersection AND degrades the K10 parent (K11)

- Expectation (H1, preregistered): CueTraceArbitrator
  (attention_cue_trace.py: per-cue gain vectors from K10 x per-cue
  eligibility traces with K9's baseline-free return-conditioned update;
  attention.py and attention_cue.py UNTOUCHED) would beat each parent arm
  on cue_delayed_reward v1.0.0 (new env: canonical delayed_reward
  mechanics + per-episode cue {0,1} in the observation; correct = cue) —
  G1 R_comb/K10-arm >= 1.15 AND G2 R_comb/frozen >= 1.30 on >= 3/4 fresh
  seeds {1212, 3434, 5656, 7878} — without degrading either parent task
  (P2: R_comb/K9 >= 0.90 on canonical delayed_reward; P3:
  R_comb/frozen >= 1.30 on cue-conditioned changing_rule).
- Observed:
  - P1 (intersection): combined 0.14-0.36 vs K10-arm 0.50-0.58 vs K9-arm
    0.10-0.18 vs frozen 0.18-0.46 (3.10 lucky on 7878). G1 R = 0.28-0.64
    (0/4), G2 R = 0.12-0.78 (0/4). The combination is WORSE than both
    parents and <= frozen on every seed.
  - P2 (K9 parent): R_comb/K9 = 1.00 EXACT on all 4 seeds (identical
    totals) — the combination with constant context is behaviorally
    identical to the K9 trace path. The module is correct; the failure
    is substantive, not a bug.
  - P3 (K10 parent): R_comb/frozen = 0.93-1.12 (0/4, DEGRADED);
    head-to-head vs K10-arm 0.58-0.70. The trace rule destroys the
    cue-adapter's proven win.
- Mechanism (measured):
  (1) Intersection: the preregistered H0 predicted per-cue forward
  saturation; the realized failure is more basic — combined per-cue gains
  stayed near-flat (1.00-1.22) because the baseline-free rule updates ONLY
  on r != 0 and the +1.0 was almost never reached (totals at shaping
  level). With no decay and no error signal, the trace rule provides no
  exploration pressure and no recovery signal; the per-cue vectors never
  separate. Cue-indexing gives the traces somewhere to attach, but the
  state-blind bandit still cannot learn branch-then-forward sequencing:
  NR-A-006 stands, and cue-indexing does not move it. (The K10 arm's
  delta rule decayed less harmfully and beat the combination here —
  0.50-0.58 — a real measured inversion of the H1 ordering.)
  (2) Parent degradation: on dense changing_rule the baseline-free trace
  rule reinforces ALL recent actions on every r=1 tick (trace smearing).
  Seed 1212 final vectors — combined cue0 {a0: 2.0, a1: 1.0}, cue1
  {a0: 2.0, a1: 1.22} (a0 smeared to cap in BOTH contexts) vs K10-arm cue0
  {a0: 1.925, a1: 0.8}, cue1 {a0: 0.65, a1: 2.0} (clean divergence). The
  constant-baseline delta rule does per-action error correction (only the
  chosen arm moves); it was load-bearing for the K10 win. The K9 rule is
  actively harmful wherever rewards are dense.
- Rules out: "adding input-conditioned gains to the trace rule lifts
  NR-A-006" and "the trace rule is a drop-in upgrade over the delta
  rule." The two mechanisms are NOT complementary: the trace rule needs
  sparse-return isolation to avoid smearing, and the cue adapter needs
  per-action error correction to diverge per-cue vectors. Composition of
  two working mechanisms produced a strictly-worse-than-either agent.
- Receipt: receipts/k11_cue_trace_combination.ndjson (hash-chained:
  preregistration + 12 seed_results + verdict; chain verified).
  Prereg: receipts/prereg_k11_cue_trace_combination.json.
- New modules: prototypes/architecture-a/attention_cue_trace.py
  (CueTraceArbitrator), flesh-pits/experiments/envs/cue_delayed_reward.py
  (CueDelayedReward v1.0.0, registered additively in envs/__init__.py;
  canonical envs untouched). Fabrication-tripwire: CLEAN (architecture-a
  tree + envs dir, pre-run).
