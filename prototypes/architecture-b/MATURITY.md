# CAUSAL MATURITY — Architecture B prototype v1

*Directive §41. Tracked honestly. 2026-10-07.*

Scale: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.

## Component maturity

| Component | Maturity | Evidence |
|---|---|---|
| HierarchicalGenerativeModel (L0 linear + L1 context experts; persistent mu0) | **INTEGRATED** | K2 survive 5/5 on fresh seeds (learned err gap 0.405–0.426, stable — prediction learning is robust); K3B (3 seeds, longer horizon): arm-A (changing_rule, 1500 train transitions) reward-channel lesion gap +0.0602, consistent 3/3 seeds (~19× K3's +0.0032) — the contingency lives in L1; |e0| gap +0.0054 grew but seed-unstable (1/3 reversed). Arm-B (delayed_reward) |e0| gap +0.0437 consistent 3/3; terminal |rerr| ≈ 0 as preregistered (hidden branch bounds it; metric underpowered, 0–1 terminal ticks in held-out). BAR intact 24.4 > lesioned 21.4 > chance 20.0. DOWNGRADED 2026-10-07 (was ADAPTIVE): K1's adaptive-control claim overturned on replication (0/5 — frozen beats learn by 0.8–10.2 on all 5 fresh seeds; the +4.6 was a lucky draw on both agent init and env stream). The learning machinery is real (K2 5/5) but not load-bearing for control on changing_rule. K3B's longer-horizon evidence is separate and stands. **K3C (2026-10-07, 4 fresh seeds, two harder tasks): NO EFFECT — the hierarchy does not generalize beyond the two K3B envs.** Task 1 (delayed_multistep, 3× horizon + junction structure): lesion gap D_e0 seed-mean −0.0470, negative 4/4 (−0.036…−0.059) — disabling L1 IMPROVES dynamics prediction. Task 2 (compositional_rule, cue_a×cue_b×action XOR): D_rerr seed-mean −0.2684, negative 4/4 (−0.217…−0.341); D_e0 −0.0097, negative 4/4 — the lesion helps on both channels. Post-hoc diagnostic (not preregistered): on phase-7 held-out (the phase the tables trained on) intact|rerr|=0.22 vs lesioned 0.51 (D=+0.29 — the context tables DO capture the XOR within-phase); on phase-8 held-out (flipped maps) intact 0.79 vs lesioned 0.57. Mechanism: the R tables memorize the old phase's contingency and predict it confidently after the flip, while L0-only degrades gracefully to chance — context-specificity is a liability under distribution shift. Scope bound: hierarchy earns its keep on changing_rule/delayed_reward only; a two-env phenomenon. **WEAKENED 2026-10-07 (repro5_EXP-AB-K3B, 5 fresh seeds): the reward-channel gap that carried K3B (+0.0602, 3/3) reverses to −0.0319 seed-mean (negative 3/5); the signal rides the |e0| channel (+0.0079, 4/5 positive) at ~2.5× K3 scale. Preregistered gate fires REPRODUCES at exactly 4/5, but the locus and magnitude are seed-fragile — the hierarchy's longer-horizon contribution is thin.** |
| PrecisionEstimator (pi from error stats) | **CAUSAL, REJECTED (both forms)** | C2 KILL (replicated 2 seeds): uniform pi=1 beats estimated pi by 5.8–9.5 return points on changing_rule. C2B (3 seeds): the ONE preregistered shift-aware variant (surprise-triggered window reset, k=4.0) also loses — shift arm 23.97 < estimated 24.10 < uniform 28.47; stationary arm 24.40 < uniform 34.00. Diagnosed: max-channel surprise detector fired 82–110×/run (expected ~1 — cannot separate rule flips from single noisy trials), and post-reset cold-start reintroduces overshoot via tiny-window pi estimates. Do not ship in either form. **C2B 5-seed replication 2026-10-07 (repro5_EXP-AB-C2B): revival fails 0/5 — shift_reset 25.60 < uniform 28.68 (shift), 26.60 < 33.28 (stationary).** |
| EpisodicStore (experience_store port + provenance) | **CAUSAL** | M1 (3 seeds): disabling the store selectively degrades pomaze (seed-mean delta +2.200; all 3 seeds +1.83…+2.39) vs delayed_reward (+0.046). Retrieval correction + per-action error prediction are load-bearing on the partial-observability task. NOT decorative. Scope-bounded: shown on pomaze/delayed_reward only, not generalizing. **REPRODUCED 2026-10-07 (repro5_EXP-AB-M1, 5 fresh seeds {75101..75105}): selective deficit 5/5, pomaze deltas +1.71…+2.80 (seed-mean +2.158), delayed_reward +0.03…+0.09 (seed-mean +0.050).** |
| consolidation_cycle race (S-01 vs memory_port, EXP-FP-0005) | **EXECUTED (negative result)** | 2026-10-07, 4 fresh seeds, pomaze corpus, all gates passed (receipts `../../receipts/EXP-FP-0005-S.json`, `-D.json`). Offline replay shows NO measured probe improvement vs no-replay: seed-mean IG P1 (S-01) +0.049, P2 (memory_port) −0.064, U (uniform) +0.323, N 0.0; G1 fails for both prioritized arms (P1-vs-N 2/4, P2-vs-N 1/4 per-seed wins). Uniform replay beats BOTH prioritized arms 4/4 — PE-magnitude prioritization actively hurts (replays noisy/outlier transitions; uniform covers the distribution). Implementation race: P1 beats P2 4/4 (Δ=+0.113 seed-mean IG) — S-01's "promote ids, replay raw" outperforms memory_port's "replay compressed summaries" on this corpus. Phase-4 battery "consolidation CAUSAL" gets a BOUND: mechanism behaves per docs, no offline performance lift measured here. First run declared VOID pre-interpretation per G0c (memory_port merge collapsed ~4549 transitions to ~17–21 entries; raw top-200 batch = 17–21 items); amended to multiplicity replay through consolidated entries, implementation untouched; voided numbers discarded. Negative result recorded in `../../research/negative_results.md`. |
| §30 next-obs / action-consequence predictions | **INTEGRATED** | Logged every tick (JSONL, `experiments_out/EXP-AB-BAR.predictions.jsonl`); consumed by action selection. **Calibration battery EXP-FP-CALIB-01 (2026-10-07, 1950 fresh ticks/domain, preregistered): MISCALIBRATED on all domains** — stated uncertainties fail to beat sequential climatology everywhere (D1 skill −0.009, D2 −0.144 under-confident, D3 −0.398 over-confident, D4 −0.056); corr(sigma,|err|) 0.02–0.17. Confidence outputs are not decision-usable as probabilities; "calibration tracked" was bookkeeping, not calibrated forecasts. Receipt: `../../receipts/EXP-FP-CALIB-01.json`. |
| §30 retrieval-usefulness prediction | **EXECUTED** | Predicted + measured every tick. Gate implemented and tested (EXP-FP-0006, 2026-10-07, 4 fresh seeds, receipt `../../receipts/EXP-FP-0006.json`): the ûhat sign-gate policy (apply iff uhat > 0) COLLAPSES to never-apply on all 4 seeds — cold-start degeneracy (uhat inits at exactly 0.0; strict > blocks from tick 0; predictor trains on realized benefit=0 and never moves). G ≡ R ≡ no-retrieval → ΔGR=0.000 exactly; preregistered rule fires NEGATIVE, but the rejection is of the gate policy, not of ûhat's ranking ability. Clean-data ranking check (arm-U logs): corr(uhat, benefit)=+0.40…+0.48, P(benefit>0)≈0.67 — the predictor learns a real ranking signal this gate cannot exploit. Gate policy REJECTED (not self-bootstrapping); predictor stays EXECUTED. Calibration battery D4 (2026-10-07): MISCALIBRATED as a probabilistic claim (observed benefit rate flat ~0.40–0.47 across predicted 0.08–0.64). |
| ActiveInferenceSelector | **INTEGRATED, label removed PERMANENTLY** | C4 KILL: greedy beat AI on preregistered IG metric (0.0502 > 0.0275). C4B (3 seeds): on the sharper uncertainty-reduction metric (probe-set prediction-error decline), AI 0.5443 < greedy 0.7124 ≈ random 0.5497 — kill confirmed permanently. The IG term's causal contribution is unproven under both metrics. Note: the probe metric rewards concentrated practice (greedy's narrow-deep experience), a caveat for future IG metric design — it does not change the verdict. **C4B 5-seed replication 2026-10-07 (repro5_EXP-AB-C4B): AI 0.4476 < greedy 0.6487 ≈ random 0.5530, kill holds 5/5.** |
| ErrorAffect (error-derived valence/arousal) | **CAUSAL, REJECTED** | K4 KILL: PAD (-0.762) beat error-affect (-0.787) on resource_world. NR-B-002: arousal saturates → permanent over-exploration. Dropped; PAD kept as baseline. |
| PadController (donor-exact baseline) | **CAUSAL** | Won K4. Kept as the cheaper baseline per the kill rule. |
| self/world distinction (EXP-SW-01-B, §9 item 12) | **CAUSAL** | 2026-10-07, 4 fresh seeds {75101–75104}, new self_world v1.0.0 env (hand SELF-caused, ball WORLD-caused, matched change statistics), open-loop learn_transition training (600 transitions), held-out predict_next eval (300 transitions). DISTINCTION DETECTED: seed-mean D_adj = +0.091 (> +0.05 gate), D_adj > 0 on 4/4 seeds (+0.019…+0.174); D_intact stable 4/4 (+0.109…+0.128) — intact mean|e| hand ≈ 0.05–0.06 vs ball ≈ 0.16–0.19. D_adj = D_intact − D_frozen on the IDENTICAL held-out stream (paired bias correction; pre-run amendment after the throwaway pilot showed the frozen arm gaps positive from position-marginal sampling noise). Interpretation: the action-conditioned generative model predicts self-caused changes better than world-caused ones — the prediction-error gap is LEARNED (frozen shows only stream noise), i.e. B carries the self/world distinction in its prediction error. Bounds: single env, open-loop training only; per-seed D_adj variance is frozen-noise dominated on 2/4 seeds. Receipt: `../../receipts/EXP-SW-01-B.json` (hash-chained). |

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
| M1 no-memory ablation | pomaze delta +2.200 (3/3 seeds), delayed_reward +0.046 | **SURVIVES — REPRODUCES 5/5 (2026-10-07)**: identical instrument on 5 fresh seeds {75101..75105}: pomaze deltas +1.71…+2.80 (seed-mean +2.158), delayed_reward +0.03…+0.09 (seed-mean +0.050), selective deficit holds 5/5. G0a determinism spot-check PASS (seed 75101 re-run through identical helper calls, all 4 numbers match to 1e-12). First arch-B component to earn REPRODUCED |
| C2B shift-robust precision | shift_reset 23.97 < estimated 24.10 < uniform 28.47 (shift); 24.40 < 34.00 (stationary) | **KILL — REVIVAL FAILS, REPLICATES 5/5 (2026-10-07)**: 5 fresh seeds {75201..75205}: shift arm shift_reset 25.60 vs estimated 24.36 vs uniform 28.68; stationary shift_reset 26.60 vs uniform 33.28 — revival rule fails on all 5 seeds (0/5 revival). The shift-aware variant is dead in every tested form |
| C4B uncertainty-reduction IG | AI 0.5443 < greedy 0.7124 ≈ random 0.5497 (probe-error decline) | **KILL CONFIRMED PERMANENTLY — REPLICATES 5/5 (2026-10-07)**: 5 fresh seeds {75301..75305}: AI IG_probe 0.4476 vs greedy 0.6487 vs random 0.5530 (seed-means); AI ≤ max(greedy, random) on all 5 seeds. The IG term is not causal under either metric. Label stays off |
| K3B longer-horizon hierarchy | arm-A reward-channel gap +0.0602 (3/3 seeds, ~19× K3); arm-B |e0| gap +0.0437 (3/3) | **SURVIVES per frozen gate (4/5) BUT MATERIALLY WEAKENED (2026-10-07)**: 5 fresh seeds {75401..75405} — per-seed "grown" holds 4/5, so the preregistered gate fires REPRODUCES. But the carrying channel REVERSED: arm-A |rerr| gap −0.0319 seed-mean, negative on 3/5 seeds (lesion helps reward prediction on 75402 −0.0305, 75403 −0.0855, 75404 −0.2753). The signal now rides the |e0| channel (+0.0079 seed-mean, positive 4/5) at only ~2.5× K3's scale, not 19×. The original's reward-channel strength was seed-fragile; the hierarchy's locus and magnitude on longer horizons are not stable. Scope bound tightens: hierarchy earns its keep on changing_rule only via the dynamics channel, and even that is thin. Recorded as a negative in research/negative_results.md |
| K3C hierarchy beyond the two envs | task1 (delayed_multistep) D_e0 seed-mean −0.0470, 0/4 positive; task2 (compositional_rule) D_rerr seed-mean −0.2684, 0/4 positive | **NO EFFECT — scope bounded**: hierarchy is a two-env phenomenon. Context tables capture XOR within-phase (phase-7 D_rerr +0.29, post-hoc) but memorize stale contingencies under shift |

## Open gaps (honest)

1. Phase-3 battery (K1–K5, C1, C2, C4, BAR) multi-seed replicated 2026-10-07
   with mixed results — K2 REPRODUCES 5/5; K1/K3/K5 KILL on replication;
   C1 MIXED (3/5); C2 KILL, FRAGILE (3/5); C4 KILL REPLICATES (4/5); BAR
   MIXED verdict reproduces 4/5. Nothing in Phase-3 earns REPRODUCED.
   Phase-4 experiments (M1, C2B, C4B, K3B) each ran 3 fresh seeds.
2. Retrieval-usefulness prediction: gate IMPLEMENTED and TESTED (EXP-FP-0006, 2026-10-07) — the sign-gate policy collapsed to never-apply on all 4 seeds (cold-start degeneracy; not self-bootstrapping) and is REJECTED as a policy. Gap closed by rejection, not integration. The predictor learns a real ranking signal (corr(uhat,benefit)=+0.40…+0.48 on clean data) that remains unexploited — exploiting it needs a non-degenerate instrument (shadow training, warm start, or non-strict threshold), a future experiment.
3. Hierarchy generalization beyond changing_rule/delayed_reward TESTED 2026-10-07 (K3C): does NOT generalize — lesion gaps vanish/go negative on delayed_multistep and compositional_rule (4/4 seeds each). The hierarchy is a two-env phenomenon; context tables are a liability under distribution shift (post-hoc phase-7/8 diagnostic).
4. Precision weighting rejected in both tested forms (estimated, shift-reset);
   no further precision variants preregistered — the idea is shelved, not
   merely paused.
5. Self/world distinction (EXP-SW-01-B, 2026-10-07): DETECTED on one env
   (self_world) under open-loop training — the prediction-error gap is
   learned and paired-bias-corrected, but closed-loop attribution and
   multi-env generalization are untested. Architecture A shows NO
   distinction on the same probe (expected negative; no action-conditioned
   path exists in A).

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

2026-10-07 — EXP-AB-5SEED: Phase-4 5-seed multi-seed replication (§9 item #3).
Identical instruments (experiments_phase4.py verbatim), 5 fresh seeds each
(M1 {75101..75105}, C2B {75201..75205}, C4B {75301..75305}, K3B
{75401..75405}). Receipts: prototypes/architecture-b/receipts/repro5_EXP-AB-*.json
(detail, K3C-style chain link to the original) + receipts/repro5_EXP-AB-*.json
(hash-chained lane receipts). Driver: repro_phase4_5seed.py (additive).
Tripwire CLEAN; gate suite green at close.
- M1: REPRODUCES 5/5 — EpisodicStore earns REPRODUCED (first arch-B
  component).
- C2B: KILL REPLICATES 5/5 (revival holds 0/5).
- C4B: KILL REPLICATES 5/5. Driver labeling bug caught at close: the verdict
  map inverted C4B's holds-count (it counts kill-condition holds, not
  revival) and wrote KILL OVERTURNED; corrected to KILL REPLICATES in both
  receipts with an amendment note, receipts re-hashed. Per-seed data
  untouched.
- K3B: preregistered gate fires REPRODUCES at exactly 4/5, but the
  reward-channel gap that carried K3B (+0.0602, 3/3) REVERSES to −0.0319
  seed-mean (negative on 3/5 seeds); the signal now rides the |e0| channel
  (+0.0079, positive 4/5) at only ~2.5× K3's scale. The +0.0602 was
  seed-fragile. Verdict stands per the frozen gate; the hierarchy's
  longer-horizon contribution is substantively weakened (scope bound
  tightened; see kill table and research/negative_results.md).
