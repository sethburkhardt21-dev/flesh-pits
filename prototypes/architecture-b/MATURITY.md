# CAUSAL MATURITY — Architecture B prototype v1

*Directive §41. Tracked honestly. 2026-10-07.*

Scale: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.

## Component maturity

| Component | Maturity | Evidence |
|---|---|---|
| HierarchicalGenerativeModel (L0 linear + L1 context experts; persistent mu0) | **INTEGRATED** | K2 survive 5/5 on fresh seeds (learned err gap 0.405–0.426, stable — prediction learning is robust); K3B (3 seeds, longer horizon): arm-A (changing_rule, 1500 train transitions) reward-channel lesion gap +0.0602, consistent 3/3 seeds (~19× K3's +0.0032) — the contingency lives in L1; |e0| gap +0.0054 grew but seed-unstable (1/3 reversed). Arm-B (delayed_reward) |e0| gap +0.0437 consistent 3/3; terminal |rerr| ≈ 0 as preregistered (hidden branch bounds it; metric underpowered, 0–1 terminal ticks in held-out). BAR intact 24.4 > lesioned 21.4 > chance 20.0. DOWNGRADED 2026-10-07 (was ADAPTIVE): K1's adaptive-control claim overturned on replication (0/5 — frozen beats learn by 0.8–10.2 on all 5 fresh seeds; the +4.6 was a lucky draw on both agent init and env stream). The learning machinery is real (K2 5/5) but not load-bearing for control on changing_rule. K3B's longer-horizon evidence is separate and stands. NOT generalizing (contingency shown on changing_rule/delayed_reward only). |
| PrecisionEstimator (pi from error stats) | **CAUSAL, REJECTED (both forms)** | C2 KILL (replicated 2 seeds): uniform pi=1 beats estimated pi by 5.8–9.5 return points on changing_rule. C2B (3 seeds): the ONE preregistered shift-aware variant (surprise-triggered window reset, k=4.0) also loses — shift arm 23.97 < estimated 24.10 < uniform 28.47; stationary arm 24.40 < uniform 34.00. Diagnosed: max-channel surprise detector fired 82–110×/run (expected ~1 — cannot separate rule flips from single noisy trials), and post-reset cold-start reintroduces overshoot via tiny-window pi estimates. Do not ship in either form. |
| EpisodicStore (experience_store port + provenance) | **CAUSAL** | M1 (3 seeds): disabling the store selectively degrades pomaze (seed-mean delta +2.200; all 3 seeds +1.83…+2.39) vs delayed_reward (+0.046). Retrieval correction + per-action error prediction are load-bearing on the partial-observability task. NOT decorative. Scope-bounded: shown on pomaze/delayed_reward only, not generalizing. |
| consolidation_cycle race (S-01 vs memory_port, EXP-FP-0005) | **EXECUTED (negative result)** | 2026-10-07, 4 fresh seeds, pomaze corpus, all gates passed (receipts `../../receipts/EXP-FP-0005-S.json`, `-D.json`). Offline replay shows NO measured probe improvement vs no-replay: seed-mean IG P1 (S-01) +0.049, P2 (memory_port) −0.064, U (uniform) +0.323, N 0.0; G1 fails for both prioritized arms (P1-vs-N 2/4, P2-vs-N 1/4 per-seed wins). Uniform replay beats BOTH prioritized arms 4/4 — PE-magnitude prioritization actively hurts (replays noisy/outlier transitions; uniform covers the distribution). Implementation race: P1 beats P2 4/4 (Δ=+0.113 seed-mean IG) — S-01's "promote ids, replay raw" outperforms memory_port's "replay compressed summaries" on this corpus. Phase-4 battery "consolidation CAUSAL" gets a BOUND: mechanism behaves per docs, no offline performance lift measured here. First run declared VOID pre-interpretation per G0c (memory_port merge collapsed ~4549 transitions to ~17–21 entries; raw top-200 batch = 17–21 items); amended to multiplicity replay through consolidated entries, implementation untouched; voided numbers discarded. Negative result recorded in `../../research/negative_results.md`. |
| §30 next-obs / action-consequence predictions | **INTEGRATED** | Logged every tick (JSONL, `experiments_out/EXP-AB-BAR.predictions.jsonl`); consumed by action selection. **Calibration battery EXP-FP-CALIB-01 (2026-10-07, 1950 fresh ticks/domain, preregistered): MISCALIBRATED on all domains** — stated uncertainties fail to beat sequential climatology everywhere (D1 skill −0.009, D2 −0.144 under-confident, D3 −0.398 over-confident, D4 −0.056); corr(sigma,|err|) 0.02–0.17. Confidence outputs are not decision-usable as probabilities; "calibration tracked" was bookkeeping, not calibrated forecasts. Receipt: `../../receipts/EXP-FP-CALIB-01.json`. |
| §30 retrieval-usefulness prediction | **EXECUTED** | Predicted + measured every tick (per-tick logs in `experiments_out/EXP-AB-BAR.predictions.jsonl`), but uhat does NOT gate retrieval (unconditional correction) → not integrated. Gap. Calibration battery D4 (2026-10-07): MISCALIBRATED — observed benefit rate flat ~0.40–0.47 across predicted 0.08–0.64; the learned predictor barely discriminates. |
| ActiveInferenceSelector | **INTEGRATED, label removed PERMANENTLY** | C4 KILL: greedy beat AI on preregistered IG metric (0.0502 > 0.0275). C4B (3 seeds): on the sharper uncertainty-reduction metric (probe-set prediction-error decline), AI 0.5443 < greedy 0.7124 ≈ random 0.5497 — kill confirmed permanently. The IG term's causal contribution is unproven under both metrics. Note: the probe metric rewards concentrated practice (greedy's narrow-deep experience), a caveat for future IG metric design — it does not change the verdict. |
| ErrorAffect (error-derived valence/arousal) | **CAUSAL, REJECTED** | K4 KILL: PAD (-0.762) beat error-affect (-0.787) on resource_world. NR-B-002: arousal saturates → permanent over-exploration. Dropped; PAD kept as baseline. |
| PadController (donor-exact baseline) | **CAUSAL** | Won K4. Kept as the cheaper baseline per the kill rule. |

## Kill-experiment outcomes

| ID | Result | Verdict |
|---|---|---|
| K1 learning freeze | learn 26.2 > frozen 21.6 (+4.6) | KILL on replication (2026-10-07): 0/5 fresh seeds hold — frozen beats learn by 0.8–10.2 on all 5. Original +4.6 was a lucky draw on both agent init and env stream (decomposition verified). Was: SURVIVES |
| K2 learned vs fixed | learned err 0.98 < fixed 1.32 | SURVIVES — learning not decorative. REPRODUCES 5/5 (2026-10-07): gap 0.405–0.426, stable |
| K3 hierarchy lesion | struct +0.0032 (lesioned worse), white-noise -0.0012 (flat) | KILL on replication (2026-10-07): structured delta −0.008…−0.028 on all 5 fresh seeds (lesion HELPS). Original |e0| probe is a bad instrument — seed-fragile, diluted. Was: SURVIVES, WEAK. (K3B's reward-channel instrument is separate and stands) |
| K4 affect vs PAD | PAD -0.762 > error -0.787 | KILL — drop error-affect, keep PAD |
| K5 shuffle | shuffled 0.8606 > ordered 0.8575 | KILL on replication (2026-10-07): shuffled-trained ≤ ordered-trained on all 5 fresh seeds (delta −0.0002…−0.0457). Temporal-prediction claim demoted per the preregistered kill condition. Was: SURVIVES, WEAK |
| C1 error decline | delayed_reward ✓, grid_world ✓ | MIXED on replication (2026-10-07): 3/5 hold; flips small-magnitude and per-env, not systematic. Was: HOLDS |
| C2 precision ablation | uniform wins by 5.8–9.5 (2 seeds) | KILL, FRAGILE (2026-10-07): uniform wins 2.5–9.9 on 3/5 fresh seeds; precision wins 0.8, 3.5 on 2. Engineering rejection stands (uniform wins more often, larger margins; C2B independently rejects the shift-aware variant), but the single-seed kill margin is seed-dependent |
| C4 active inference | greedy IG > AI IG | KILL REPLICATES 4/5 (2026-10-07; 1 flip: seed 72602, AI IG 0.1082 > greedy 0.0656). C4B confirms permanently — label stays off |
| BAR concrete bar | intact 24.4 > lesioned 21.4 > 20.0 chance | MIXED — "NOT CLEARED" reproduces 4/5 (2026-10-07; 1 seed cleared: 72704). Prereg decline metric confounded by flip; within-phase + return evidence supports hierarchy |
| M1 no-memory ablation | pomaze delta +2.200 (3/3 seeds), delayed_reward +0.046 | SURVIVES — selective deficit; memory causally load-bearing on partial-observability task |
| C2B shift-robust precision | shift_reset 23.97 < estimated 24.10 < uniform 28.47 (shift); 24.40 < 34.00 (stationary) | KILL — revival failed; shift-aware variant joins the rejected list |
| C4B uncertainty-reduction IG | AI 0.5443 < greedy 0.7124 ≈ random 0.5497 (probe-error decline) | KILL CONFIRMED PERMANENTLY — label stays off |
| K3B longer-horizon hierarchy | arm-A reward-channel gap +0.0602 (3/3 seeds, ~19× K3); arm-B |e0| gap +0.0437 (3/3) | SURVIVES, STRENGTHENED — hierarchy earns its keep on longer horizons |

## Open gaps (honest)

1. Phase-3 battery (K1–K5, C1, C2, C4, BAR) multi-seed replicated 2026-10-07
   with mixed results — K2 REPRODUCES 5/5; K1/K3/K5 KILL on replication;
   C1 MIXED (3/5); C2 KILL, FRAGILE (3/5); C4 KILL REPLICATES (4/5); BAR
   MIXED verdict reproduces 4/5. Nothing in Phase-3 earns REPRODUCED.
   Phase-4 experiments (M1, C2B, C4B, K3B) each ran 3 fresh seeds.
2. Retrieval-usefulness prediction not wired to gate retrieval.
3. Hierarchy generalization beyond changing_rule/delayed_reward untested.
4. Precision weighting rejected in both tested forms (estimated, shift-reset);
   no further precision variants preregistered — the idea is shelved, not
   merely paused.

CONSCIOUSNESS: UNRESOLVED — mechanisms, not narratives.

## REPRODUCIBILITY — multi-seed reproduction (2026-10-07)

Method: original helper functions reused from `experiments.py`
(`run_closed_loop`, `run_replay`, `collect_transitions`, `_train_eval`,
`make_agent`); experiment logic = the original code with hardcoded seeds
replaced by a fresh-seed parameter — ALL RNG streams (env episodes via
`derive_seed`, agent init/reset, white-noise data, shuffle RNG, train/eval
agent seeds) derive from the fresh primary seed. Fresh seeds distinct from
the originals (101–109). Verdict REPRODUCES if the original verdict holds
on ≥4/5 seeds. Receipts: `receipts/repro_EXP-AB-*.json`.
Driver: `repro_multiseed.py`.

Faithfulness checks (driver vs originals, run 2026-10-07): original
configs rerun through the driver reproduce the original numbers EXACTLY —
K1 +4.60, K3 +0.0032/−0.0012, K5 +0.0030. The flips below are genuine seed
effects, not driver artifacts. (The two sanity reruns rewrote
EXP-AB-K3.json/EXP-AB-K5.json; both were restored with original
`written_utc` timestamps — byte-identical content.)

| Experiment | Original verdict | Fresh seeds | Holds | Result |
|---|---|---|---|---|
| K1 learning freeze | SURVIVES (+4.6) | 72001–72005 | 0/5 | **KILL CONDITION MET** — frozen beats learn by 0.8–10.2 on all 5 seeds |
| K2 learned vs fixed | SURVIVES (gap 0.34) | 72101–72105 | 5/5 | **REPRODUCES** (gap 0.405–0.426, stable) |
| K3 hierarchy lesion | SURVIVES, WEAK (+0.0032) | 72201–72205 | 0/5 | **KILL on replication** — structured delta −0.008…−0.028 (lesion HELPS) all 5 seeds |
| K5 shuffle | SURVIVES, WEAK (+0.0030) | 72301–72305 | 0/5 | **KILL on replication** — shuffled-trained ≤ ordered-trained all 5 seeds (delta −0.0002…−0.0457) |
| C1 error decline | HOLDS | 72401–72405 | 3/5 | **MIXED** — flips small-magnitude (grid_world −0.0075 on 72403; delayed_reward −0.0235 on 72405) |
| C2 precision ablation | KILL (uniform wins 5.8–9.5) | 72501–72505 | 3/5 | **KILL FRAGILE** — uniform wins 2.5–9.9 on 3 seeds; precision wins 0.8, 3.5 on 2 seeds |
| C4 active inference | KILL (greedy IG wins) | 72601–72605 | 4/5 | **KILL REPLICATES** (1 flip: seed 72602, AI IG 0.1082 > greedy 0.0656) |
| BAR concrete bar | MIXED (NOT CLEARED) | 72701–72705 | 4/5 | **REPRODUCES** — "NOT CLEARED" holds (1 seed cleared: 72704) |

K1 decomposition (isolating the seed-fragility source):
- primary=101, agent=11 (ORIGINAL config): delta +4.60 — reproduces original
- primary=101, agent=72001 (agent init seed ONLY changed): delta −0.20 — kill
- primary=72001, agent=11 (env stream ONLY changed): delta −5.60 — kill
The +4.6 was a lucky draw on BOTH the agent init and the env stream. The
frozen agent's random-init weights + mu0 tracking do most of the work;
learning updates are not robustly load-bearing for control on changing_rule.

Maturity changes from this reproduction:
- HierarchicalGenerativeModel: ADAPTIVE → **INTEGRATED**. K1's
  adaptive-control claim is overturned (0/5); the learning machinery is
  real (K2 5/5: prediction learning robust) but not load-bearing for
  control. This demotion covers the ORIGINAL K3 probe verdict and K1 only —
  K3B's longer-horizon evidence (reward-channel gap +0.0602, 3/3 seeds) is
  separate and stands.
- K3 original probe: SURVIVES, WEAK → **KILL on replication**. The |e0|
  probe is a bad instrument (seed-fragile, diluted by unpredictable
  channels); the hierarchy signal lives in the reward-channel/return
  instruments (K3B, BAR).
- K5: SURVIVES, WEAK → **KILL (demoted per the preregistered kill
  condition)** — temporal-prediction claim demoted.
- C1: HOLDS → **MIXED**.
- C2: KILL → **KILL, FRAGILE (3/5)**. The rejection stands as the
  engineering call — and C2B independently rejects the shift-aware variant —
  but the original single-seed kill margin is seed-dependent.
- C4: KILL → **KILL REPLICATES (4/5)**; C4B confirms permanently.
- BAR: MIXED → **MIXED (verdict reproduces 4/5)**.
- EpisodicStore, ErrorAffect, PadController: unchanged (outside this
  reproduction's mandate; M1 independently promoted EpisodicStore).

Cross-cutting pattern: the failures cluster exactly where L1 / precision /
learning interact with distribution shift — K1 (freeze helps), K3 (lesion
helps), C2 (uniform wins). Pure prediction learning (K2) is robust 5/5.
Coherent reading: the predictor LEARNS, but the L1-context and precision
machinery does not robustly convert learning into better decisions under
shift. Phase-4 follow-up should test this directly.

Open gap #1 (single seeds) is now: Phase-3 battery multi-seed with mixed
results (above); nothing in Phase-3 earns REPRODUCED. Phase-4 experiments
(M1, C2B, C4B, K3B) each ran 3 fresh seeds per the concurrent worker.
