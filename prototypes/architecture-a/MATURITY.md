# ARCHITECTURE A — CAUSAL MATURITY LEDGER

*Directive §41. Updated 2026-10-07 after K1–K4. Levels: PRESENT →
EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.
Marked honestly; nothing above its evidence.*

| Component | Level | Evidence |
|---|---|---|
| `workspace_buffer.py` (bounded buffer, eviction policy) | **REPRODUCED** | K1: interference 0→0.507→1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf. Lesioning capacity causally changes competition. Multi-seed reproduction 5/5 fresh seeds, verdict holds every seed (receipt `receipts/repro_k1_capacity_lesion.json`). Promoted 2026-10-07. Not ADAPTIVE (K, policy fixed). |
| `bids.py` (RunningZScoreBid: change + habituation) | **CAUSAL** | K7: experiment-side lesion (constant-0.5 stubs) collapses arbitration winner-entropy 1.03–1.38 → exactly 0.0 bits on 4/4 seeds — arbitration is completely changed by the lesion. BOUND (NR-A-009): the preregistered reward proxy failed (T=1.14–1.25 < 1.30, 0/4) — fixed-winner reward buffering + learned-gain compensation (lesioned+learned 128–130 ≈ intact+learned 139–140). Precise claim: bids are causal for moment-to-moment arbitration under frozen gains; the learned system barely needs them. |
| `attention.py` (arbitration + learned gains) | **GENERALIZING** | K4: freezing gains changes total reward 1.57–2.03× (CAUSAL). Gains move with utility history via broadcast feedback (gain_history trail = ADAPTIVE). K5: win generalizes to noisy-signal tracking (P2: 4/4, R 1.31–1.52) and multi-reversal stationary shifts (P3: 4/4, R 1.74–2.05). BOUNDS: NR-A-006 (delayed_reward: learned LOSES, R 0.08–0.17 — constant-baseline delta rule decays all gains under sparse reward); NR-A-007 (cue-conditioned changing_rule without cue input: no gap, R 1.03–1.09); NR-A-005 (signal-tracking: frozen change-bids suffice). Not REPRODUCED (single lab, no independent replication). |
| `ignition.py` (recurrent bistable gate) | **REPRODUCED** | K3: hard Heaviside step, W=0.100; gate causally determines propagation. Multi-seed reproduction 5/5 fresh seeds (K3 probe W 0.08–0.10 < 0.12; receipt `receipts/repro_k3_ignition_probe.json`). NR-A-004 RESOLVED and WIRED (K6 + K8, 2026-10-07): R1 sub-ignition exploratory path is now the DEFAULT `_select_action` — when nothing ignites, act on the graded arbitration winner without propagating anything to consumers (gate still decides ALL propagation; K2 sole-path unaffected). K8 (fresh seeds 51501–51503): frozen-gains ignition trajectories identical 3/3 on behavioral content (ignited item id+strength, bids, winner — NR-A-010 documents the item_id harness artifact; gate selectivity preserved mechanismally); zero non-feedback consumer deliveries on all sub-ignition trials (19–21/seed); phase-2 'c' fraction 0.0→0.79–0.81, total 84–85→128–129; K4 non-degradation 4/4 (R 1.57–2.03). Post-wire K1–K4 reruns byte-identical (receipt `receipts/k8_r1_wiring_decision.json`, post_wire_reruns; at theta=0.45 the path never fires in those scenarios — exploratory behavior activates only in the sub-ignition regime). R2 (stationarity detector lowering theta) REJECTED: admits ~2.7× ignitions, weakening gate selectivity. Not ADAPTIVE (theta/gain/feedback fixed). |
| `broadcast.py` (sole-path bus) | **REPRODUCED** | K2: lesion_all → all six consumers silent at once; internal processing continues. No side channels detected. Multi-seed reproduction 5/5 fresh seeds (receipt `receipts/repro_k2_broadcast_lesion.json`). Promoted 2026-10-07. |
| consumers: memory_admit, self_model_update, planner_input, consolidation_eligible, report | **REPRODUCED** | K2: each consumer's instrumented effect appears/vanishes exactly with its broadcast deliveries. Multi-seed reproduction 5/5 fresh seeds (receipt `receipts/repro_k2_broadcast_lesion.json`). Promoted 2026-10-07. |
| consumer: attention_update | **ADAPTIVE** | Closes the K4 learning loop; gains move only through its broadcast feedback. |
| `tick.py` (sole-path wiring, closed loop) | **INTEGRATED** | Full pipeline executes closed-loop; broadcast verified as sole content path (K2). Whole-tick CAUSAL not separately claimed. 2026-10-07: `_select_action` hook added for NR-A-004 candidate testing — proven behavior-identical (K4 rerun bit-identical; K1–K3 reruns byte-identical, all PASS). 2026-10-07 (K8 decision: WIRE): R1 sub-ignition exploratory path is now the default `_select_action` (no-proposal → act on graded arbitration winner, zero consumer propagation). Post-wire K1–K4 reruns byte-identical (receipt `receipts/k8_r1_wiring_decision.json`, post_wire_reruns; sha-verified 2026-10-07 by the maturity audit), all verdicts PASS — behavior preserved except the intended exploratory actions, which fire only in the sub-ignition regime. |
| `envs.py` (ChangingRelevanceEnv) | **EXECUTED** | Provisional local env (superseded for K4 by the canonical ENV_INTERFACE v1.0.0; K4 re-run PASS 4/4 against canonical changing_rule — receipt flesh-pits/receipts/K4-CANONICAL-RERUN.ndjson). |
| identity symmetry (§§14–15) | **CAUSAL** (test) | Bit-identical trajectories for novel labels incl. a historically privileged string; tie-break is label-agnostic channel order. |

## What failed or is BLOCKED

- **Bid-ablation for CAUSAL bids**: DONE (K7, 2026-10-07) — bids promoted
  to CAUSAL for arbitration dynamics, with the NR-A-009 bound (reward
  impact modest; learned gains compensate).
- **Generalization of learned attention**: EXTENDED (K5, 2026-10-07) —
  generalizes to noisy-signal tracking and multi-reversal stationary
  shifts; bounded by NR-A-006 (delayed_reward) and NR-A-007
  (cue-conditioned changing_rule). Attention promoted to GENERALIZING
  with stated bounds.
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
