# ARCHITECTURE A — CAUSAL MATURITY LEDGER

*Directive §41. Updated 2026-10-07 after K1–K4. Levels: PRESENT →
EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.
Marked honestly; nothing above its evidence.*

| Component | Level | Evidence |
|---|---|---|
| `workspace_buffer.py` (bounded buffer, eviction policy) | **REPRODUCED** | K1: interference 0→0.507→1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf. Lesioning capacity causally changes competition. Multi-seed reproduction 5/5 fresh seeds, verdict holds every seed (receipt `receipts/repro_k1_capacity_lesion.json`). Promoted 2026-10-07. Not ADAPTIVE (K, policy fixed). |
| `bids.py` (RunningZScoreBid: change + habituation) | **CAUSAL** | K7: experiment-side lesion (constant-0.5 stubs) collapses arbitration winner-entropy 1.03–1.38 → exactly 0.0 bits on 4/4 seeds — arbitration is completely changed by the lesion. BOUND (NR-A-009): the preregistered reward proxy failed (T=1.14–1.25 < 1.30, 0/4) — fixed-winner reward buffering + learned-gain compensation (lesioned+learned 128–130 ≈ intact+learned 139–140). Precise claim: bids are causal for moment-to-moment arbitration under frozen gains; the learned system barely needs them. |
| `attention.py` (arbitration + learned gains) | **GENERALIZING** | K4: freezing gains changes total reward 1.57–2.03× (CAUSAL). Gains move with utility history via broadcast feedback (gain_history trail = ADAPTIVE). K5: win generalizes to noisy-signal tracking (P2: 4/4, R 1.31–1.52) and multi-reversal stationary shifts (P3: 4/4, R 1.74–2.05). K9 (2026-10-07): return-conditioned baseline-free eligibility-trace redesign (`attention_sparse.py`) FAILED to lift the sparse-reward bound — R 0.34–0.50, 0/4 fresh seeds; NR-A-011 recorded. The bound is structural: a state-blind bandit cannot learn branch-then-forward sequencing (learned forward preference poisons the t=0 no-op decision; the +1.0 is too rare for trace credit to matter). K10 (2026-10-07): NR-A-007 CONDITIONALLY LIFTED — cue-indexed gains (`attention_cue.py`, original delta rule unchanged) beat frozen 4/4 (R 1.52–1.68) on cue-conditioned changing_rule WITH cue input; control (cue withheld, constant context) replicates the bound (R 0.95–1.13, 0/4). Per-cue gain vectors diverged as designed (mechanism evidence in receipt). Current bounds: NR-A-006/NR-A-011 (sparse delayed reward — structural); NR-A-007-without-cue-input (architectural input bound, replicates 0/4); NR-A-005 (signal-tracking: frozen change-bids suffice). Second-lane independent replication (2026-10-07, fresh seeds, drivers rewritten from spec, preregs sealed before drivers): K9 REPRODUCES 0/4 (R 0.24–0.67, seeds 91511–91514; receipt receipts/repro_second_lane_k9.ndjson) — NR-A-011 holds. K10 REPRODUCES: B1 4/4 (R 1.45–1.58, seeds 91615–91618), B2 control 0/4 gaps (R 0.88–1.07; receipt receipts/repro_second_lane_k10.ndjson) — NR-A-007 conditional lift holds. Maturity of the K9/K10 verdicts: REPRODUCED. K11 (2026-10-07): (trace-rule x cue-adapter) combination (`attention_cue_trace.py`; new `cue_delayed_reward` v1.0.0 env) FAILED the intersection probe 0/4 on both gates (combined 0.14-0.36 vs K10-arm 0.50-0.58, frozen 0.18-0.46; per-cue gains stayed near-flat 1.00-1.22) AND degraded the K10 parent task (R_comb/frozen 0.93-1.12, 0/4; trace smearing pushed a0 to the 2.0 cap in BOTH per-cue contexts vs K10's clean divergence) -> NR-A-012. The two mechanisms are not complementary: the trace rule needs sparse-return isolation to avoid smearing; the cue adapter needs per-action error correction. P2 equivalence control: combined-with-constant-context is behaviorally identical to K9 (R=1.00 exact, 4/4) — the failure is substantive, not a bug. Not REPRODUCED (single lab, no independent replication). |
| `ignition.py` (recurrent bistable gate) | **REPRODUCED** | K3: hard Heaviside step, W=0.100; gate causally determines propagation. Multi-seed reproduction 5/5 fresh seeds (K3 probe W 0.08–0.10 < 0.12; receipt `receipts/repro_k3_ignition_probe.json`). NR-A-004 RESOLVED and WIRED (K6 + K8, 2026-10-07): R1 sub-ignition exploratory path is now the DEFAULT `_select_action` — when nothing ignites, act on the graded arbitration winner without propagating anything to consumers (gate still decides ALL propagation; K2 sole-path unaffected). K8 (fresh seeds 51501–51503): frozen-gains ignition trajectories identical 3/3 on behavioral content (ignited item id+strength, bids, winner — NR-A-010 documents the item_id harness artifact; gate selectivity preserved mechanismally); zero non-feedback consumer deliveries on all sub-ignition trials (19–21/seed); phase-2 'c' fraction 0.0→0.79–0.81, total 84–85→128–129; K4 non-degradation 4/4 (R 1.57–2.03). Post-wire K1–K4 reruns byte-identical (receipt `receipts/k8_r1_wiring_decision.json`, post_wire_reruns; at theta=0.45 the path never fires in those scenarios — exploratory behavior activates only in the sub-ignition regime). R2 (stationarity detector lowering theta) REJECTED: admits ~2.7× ignitions, weakening gate selectivity. Not ADAPTIVE (theta/gain/feedback fixed). |
| `broadcast.py` (sole-path bus) | **REPRODUCED** | K2: lesion_all → all six consumers silent at once; internal processing continues. No side channels detected. Multi-seed reproduction 5/5 fresh seeds (receipt `receipts/repro_k2_broadcast_lesion.json`). Promoted 2026-10-07. |
| consumers: memory_admit, self_model_update, planner_input, consolidation_eligible, report | **REPRODUCED** | K2: each consumer's instrumented effect appears/vanishes exactly with its broadcast deliveries. Multi-seed reproduction 5/5 fresh seeds (receipt `receipts/repro_k2_broadcast_lesion.json`). Promoted 2026-10-07. |
| consumer: attention_update | **ADAPTIVE** | Closes the K4 learning loop; gains move only through its broadcast feedback. |
| `tick.py` (sole-path wiring, closed loop) | **INTEGRATED** | Full pipeline executes closed-loop; broadcast verified as sole content path (K2). Whole-tick CAUSAL not separately claimed. 2026-10-07: `_select_action` hook added for NR-A-004 candidate testing — proven behavior-identical (K4 rerun bit-identical; K1–K3 reruns byte-identical, all PASS). 2026-10-07 (K8 decision: WIRE): R1 sub-ignition exploratory path is now the default `_select_action` (no-proposal → act on graded arbitration winner, zero consumer propagation). Post-wire K1–K4 reruns byte-identical (receipt `receipts/k8_r1_wiring_decision.json`, post_wire_reruns; sha-verified 2026-10-07 by the maturity audit), all verdicts PASS — behavior preserved except the intended exploratory actions, which fire only in the sub-ignition regime. |
| `envs.py` (ChangingRelevanceEnv) | **EXECUTED** | Provisional local env (superseded for K4 by the canonical ENV_INTERFACE v1.0.0; K4 re-run PASS 4/4 against canonical changing_rule — receipt flesh-pits/receipts/K4-CANONICAL-RERUN.ndjson). |
| identity symmetry (§§14–15) | **CAUSAL** (test) | Bit-identical trajectories for novel labels incl. a historically privileged string; tie-break is label-agnostic channel order. |
| self/world distinction (EXP-SW-01-A, §9 item 12) | **EXECUTED (negative result)** | 2026-10-07, 4 fresh seeds {75101–75104}, new self_world v1.0.0 env (hand SELF-caused, ball WORLD-caused, matched change statistics). NO internal variable distinguishes matched self/world changes: seed-mean Delta_bid = +0.0009 (rule: abs(Delta) > 0.05), Delta_ign = 0.000 — nothing ever ignited (habituated bids sit below theta on both channels); gains floored symmetric (0.01/0.01). Mechanism: specialists/arbitrator/ignition have no action-conditioned path (no efference copy); the z-score bid habituates per channel to 0.5 regardless of cause. Instrument validated by unit test (chain discriminates unequal stimuli). Receipt `../../receipts/EXP-SW-01-A.json` (hash-chained). Negative recorded in `../../research/negative_results.md`. |

## What failed or is BLOCKED

- **Bid-ablation for CAUSAL bids**: DONE (K7, 2026-10-07) — bids promoted
  to CAUSAL for arbitration dynamics, with the NR-A-009 bound (reward
  impact modest; learned gains compensate).
- **Generalization of learned attention**: EXTENDED (K5, K10, 2026-10-07) —
  generalizes to noisy-signal tracking, multi-reversal stationary shifts,
  and (K10) cue-conditioned changing_rule WITH cue input (R 1.52–1.68,
  4/4; per-cue gain vectors diverge as designed). Bounded by NR-A-006 /
  NR-A-011 (delayed_reward: structural — K9's trace-rule redesign failed
  0/4, R 0.34–0.50; a state-blind bandit cannot learn branch-then-forward
  sequencing), NR-A-012 (K11: the trace-rule x cue-adapter combination
  failed the cue-conditioned delayed_reward intersection 0/4 on both
  gates AND degraded the K10 parent task 0/4 — the two mechanisms are
  not complementary: the trace rule needs sparse-return isolation, the
  cue adapter needs per-action error correction), NR-A-007-without-cue-input (architectural input bound;
  K10 control replicates 0/4, R 0.95–1.13), and NR-A-005
  (signal-tracking: frozen change-bids suffice). Attention stays
  GENERALIZING with stated bounds.
- **theta=0.6 stationarity perseveration**: RESOLVED and WIRED (K6 + K8,
  2026-10-07) — R1 sub-ignition exploratory path is the default tick
  behavior (act on the graded arbitration winner when nothing ignites;
  nothing unignited reaches consumers; gate still decides all
  propagation). K8 confirmation on fresh seeds 51501–51503: 5/5 gates
  PASS. R2 documented as the rejected alternative.
- **Canonical environment**: RESOLVED 2026-10-07 — ENV_INTERFACE.md v1.0.0 STABLE landed; K4 re-run COMPLETED against the canonical changing_rule by the test-battery worker: PASS 4/4 (ratios 1.44/1.60/1.70/1.48 ≥ 1.30; receipt flesh-pits/receipts/K4-CANONICAL-RERUN.ndjson, hash-chained). Multi-seed reproduction added 5/5 fresh seeds (repro_k4_attention_baseline.json, ratios 1.71–2.01). The "pending" note below was stale.
- Kuramoto/komplex binding, coherence-gated attention: NOT built —
  per the candidates doc they are EXPERIMENT_AGAINST, and nothing in
  K1–K4 needed them. They stay out unless a lesion demands them.

## REPRODUCIBILITY — multi-seed reproduction (2026-10-07)

Method: the reproduction worker ran the ORIGINAL experiment code paths
(`experiments/repro_multiseed_a.py` imports the original K1–K4 modules; no
copies) on 5 fresh seeds each, distinct from the originals
(K1/K2/K3 original seed 20261007; K4 original seeds 11/22/33/44).
Per-seed verdicts use the ORIGINAL preregistered gates. A kill/experiment
REPRODUCES if the verdict holds on ≥4/5 seeds.
Receipts: `receipts/repro_k1_capacity_lesion.json`,
`repro_k2_broadcast_lesion.json`, `repro_k3_ignition_probe.json`,
`repro_k4_attention_baseline.json`.

| Experiment | Fresh seeds | Per-seed verdicts | Result |
|---|---|---|---|
| K1 capacity lesion | 71001–71005 | 5/5 PASS (I@0.85 = 0.450–0.537, monotonic every seed) | **REPRODUCES** |
| K2 broadcast lesion | 71101–71105 | 5/5 PASS (selective + total + recovery every seed) | **REPRODUCES** |
| K3 ignition probe | 71201–71205 | 5/5 PASS (W = 0.08–0.10 < 0.12; p_below=0, p_above=1 every seed) | **REPRODUCES** |
| K4 attention baseline | 71301–71305 | 5/5 WIN (R = 1.71–2.13 ≥ 1.30 every seed) | **REPRODUCES** |

Maturity changes from this reproduction:
- `workspace_buffer.py`: CAUSAL → **REPRODUCED**
- `broadcast.py` + all six consumers: CAUSAL → **REPRODUCED**
- `ignition.py`: CAUSAL → **REPRODUCED**
- `attention.py`: K4 leg independently replicated 5/5 fresh seeds (R
  1.71–2.13); supports the standing GENERALIZING claim — bounds unchanged
  (NR-A-006/NR-A-007/NR-A-005).
- `attention_update` consumer, `tick.py`, `bids.py`, `envs.py`, identity
  symmetry: unchanged by this reproduction (not re-run, or already at earned
  levels via concurrent work).

## REPRODUCIBILITY — second-lane independent replication (2026-10-07)

The REPRODUCED bar's missing half, closed. An independent lane
(preregistered 2026-10-07T07:49:00Z, sealed BEFORE the run —
`receipts/prereg_repro_second_lane.json`, hash `756966a5...567728`)
rewrote all four experiment drivers from the K1–K4 spec (no import of
the original experiment scripts or the first-lane repro driver; only
the architecture modules under test) and ran them against the CURRENT
default tick (R1 sub-ignition exploratory path wired as default since
K8). 5 fresh seeds per experiment, all distinct from every prior seed
(K1: 72001–72005; K2: 72101–72105; K3: 72201–72205; K4: 72301–72305).
Driver: `experiments/repro_second_lane_a.py`. Receipts:
`receipts/repro_second_lane_k{1,2,3,4}_*.json`, hash-chained
prereg → k1 → k2 → k3 → k4 (chain verified intact).

| Experiment | Fresh seeds | Per-seed verdicts | Result |
|---|---|---|---|
| K1 capacity lesion | 72001–72005 | 5/5 PASS (I@0.85 = 0.458–0.522, monotonic every seed) | **REPRODUCES** |
| K2 broadcast lesion | 72101–72105 | 5/5 PASS (selective + total + recovery every seed) | **REPRODUCES** |
| K3 ignition probe | 72201–72205 | 5/5 PASS (W = 0.10 < 0.12; p_below=0, p_above=1; lin_err ≈ 0.0015–0.0018 every seed) | **REPRODUCES** |
| K4 attention baseline | 72301–72305 | 5/5 WIN (R = 1.46–1.88 ≥ 1.30 every seed) | **REPRODUCES** |

R1-active observations (recorded, not gated — preregistered):
- K1/K3 do not use the tick; R1 cannot fire. No behavioral difference
  possible, none observed.
- K2: sub-ignition path fired 0/30 in baseline, selective-lesion, and
  recovery phases, and 30/30 in the total-lesion phase on every seed —
  during the total lesion the planner queue stays empty (lesioned), so
  action selection falls back to the graded arbitration winner. Zero
  consumer deliveries in that phase on every seed: the R1 default
  behaves exactly as wired (K8) — action without propagation.
- K4: 0 sub-ignition actions in learned AND frozen conditions on all
  seeds (at theta=0.45 ignition occurs every tick); learned gains
  converged with 'c' at the 2.0 cap on all 5 seeds after the reversal.

Maturity changes from this second-lane replication (2026-10-07):
- `workspace_buffer.py`: stays **REPRODUCED** — now confirmed by two
  independent lanes (first: 71001–71005; second: 72001–72005).
- `broadcast.py` + all six consumers: stay **REPRODUCED** — confirmed
  by two independent lanes (71101–71105; 72101–72105), including the
  R1-active total-lesion behavior.
- `ignition.py`: stays **REPRODUCED** — confirmed by two independent
  lanes (71201–71205; 72201–72205).
- `attention.py`: K4 leg now independently replicated TWICE
  (first lane R 1.71–2.13; second lane R 1.46–1.88) — stays
  **GENERALIZING** because the K5 generalization legs (P2/P3) remain
  single-lab; bounds unchanged (NR-A-006/NR-A-007/NR-A-005).
- `tick.py`: the K8 R1-default wiring survived a second independent
  replication intact (K2/K4 verdicts unchanged; sub-ignition behavior
  matches the wired spec) — stays INTEGRATED (whole-tick CAUSAL not
  separately claimed).
- No demotions. The "single-lab" qualifier on the workspace machinery's
  REPRODUCED claims is now RESOLVED.

## 2026-10-07 — EXP-FP-0010 pomaze diagnosis: A = REPRESENTATION (extends NR-A-006)

Cross-architecture pomaze diagnostic (6 fresh seeds 91001–91006; receipt
receipts/EXP-FP-0010-POMAZE-DIAG.ndjson). A-STD: mean −4.05, p_goal 0.044 —
indistinguishable from uniform random (−4.15, 0.006). Measured:
- State-blind by construction, confirmed: mean |corr(stimulus, beacon)| =
  0.046 across 360 channel-episodes (the preregistered max rule at 0.222
  was over-strict; the mean is the informative statistic).
- NR-A-006 decay on pomaze: gains at the 0.01 floor in 98.9% of
  channel-episodes; the constant −0.01/−0.02 punishment sits below the
  0.5 baseline so the delta rule can only punish.
- With gains floored, arbitration = argmax of habituated noise bids
  (NOT tie-break: tie frac 0.000) → behavior ≈ random walk
  (stuck 0.535 vs random 0.540).
- Dense shaping (+0.02×beacon): NO navigation gain (p_goal 0.033 vs
  0.044); the +1.64 return lift is the shaping term collected while
  wandering — a state-blind policy cannot exploit a state gradient.
- NEAR (p_goal 0.744): gains leave the floor (0.789), goal rate
  0.711→0.778 second-half — the delta rule responds to frequent reward.
  But DEMO (5 NEAR then 15 canonical): ZERO transfer (p_goal 0.056 vs
  0.044; gains re-floor 0.986). A bandit cannot retain maze-navigation
  policy across maze seeds.
- Bound: on pomaze-class tasks A's failure is REPRESENTATION, not
  exploration or credit assignment. `attention.py` stays GENERALIZING;
  the NR-A-006 STRUCTURAL bound now covers pomaze explicitly.
