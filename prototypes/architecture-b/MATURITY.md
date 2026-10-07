# CAUSAL MATURITY — Architecture B prototype v1

*Directive §41. Tracked honestly. 2026-10-07.*

Scale: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.

## Component maturity

| Component | Maturity | Evidence |
|---|---|---|
| HierarchicalGenerativeModel (L0 linear + L1 context experts; persistent mu0) | **INTEGRATED** | K2 survive 5/5 on fresh seeds (learned err gap 0.405–0.426, stable — prediction learning is robust); K3B (3 seeds, longer horizon): arm-A (changing_rule, 1500 train transitions) reward-channel lesion gap +0.0602, consistent 3/3 seeds (~19× K3's +0.0032) — the contingency lives in L1; |e0| gap +0.0054 grew but seed-unstable (1/3 reversed). Arm-B (delayed_reward) |e0| gap +0.0437 consistent 3/3; terminal |rerr| ≈ 0 as preregistered (hidden branch bounds it; metric underpowered, 0–1 terminal ticks in held-out). BAR intact 24.4 > lesioned 21.4 > chance 20.0. DOWNGRADED 2026-10-07 (was ADAPTIVE): K1's adaptive-control claim overturned on replication (0/5 — frozen beats learn by 0.8–10.2 on all 5 fresh seeds; the +4.6 was a lucky draw on both agent init and env stream). The learning machinery is real (K2 5/5) but not load-bearing for control on changing_rule. K3B's longer-horizon evidence is separate and stands. **K3C (2026-10-07, 4 fresh seeds, two harder tasks): NO EFFECT — the hierarchy does not generalize beyond the two K3B envs.** Task 1 (delayed_multistep, 3× horizon + junction structure): lesion gap D_e0 seed-mean −0.0470, negative 4/4 (−0.036…−0.059) — disabling L1 IMPROVES dynamics prediction. Task 2 (compositional_rule, cue_a×cue_b×action XOR): D_rerr seed-mean −0.2684, negative 4/4 (−0.217…−0.341); D_e0 −0.0097, negative 4/4 — the lesion helps on both channels. Post-hoc diagnostic (not preregistered): on phase-7 held-out (the phase the tables trained on) intact|rerr|=0.22 vs lesioned 0.51 (D=+0.29 — the context tables DO capture the XOR within-phase); on phase-8 held-out (flipped maps) intact 0.79 vs lesioned 0.57. Mechanism: the R tables memorize the old phase's contingency and predict it confidently after the flip, while L0-only degrades gracefully to chance — context-specificity is a liability under distribution shift. Scope bound: hierarchy earns its keep on changing_rule/delayed_reward only; a two-env phenomenon. **WEAKENED 2026-10-07 (repro5_EXP-AB-K3B, 5 fresh seeds): the reward-channel gap that carried K3B (+0.0602, 3/3) reverses to −0.0319 seed-mean (negative 3/5); the signal rides the |e0| channel (+0.0079, 4/5 positive) at ~2.5× K3 scale. Preregistered gate fires REPRODUCES at exactly 4/5, but the locus and magnitude are seed-fragile — the hierarchy's longer-horizon contribution is thin.** **CHARACTERIZED 2026-10-07 (EXP-AB-K3D, 4 fresh seeds {76101..76104}, preregistered before run; receipts `../../receipts/EXP-AB-K3D.json` + `receipts/EXP-AB-K3D.json`, both hash-verified, G0b determinism PASS):** the |e0| channel is a two-env phenomenon -- NOT dead. delayed_reward: D_e0 +0.0986 seed-mean, 4/4 positive, 95pct t-CI [+0.0033, +0.1939] (excludes 0) -- the gap lives in the branch channel (+0.0504, 4/4; branch is constant within an episode; 3 L1 contexts = one per branch value memorize per-context constants that L0's global map compromises on). changing_rule: stable-phase +0.0070 (3/4, CI includes 0), post-flip +0.0103 (3/4, CI includes 0) -- the preregistered MEMORIZATION signature (advantage vanishes/reverses post-flip) does NOT fire; DYNAMICS fires by the frozen rule, carried seed-consistently by the rule-invariant last_action channel (+0.0190 post-flip, 4/4 both segments; cue channel ~0). delayed_multistep: -0.0555 (0/4, CI [-0.0765, -0.0345] excludes 0 -- the lesion HELPS; stage channel -0.0344, 4/4: 9 contexts fragment deterministic global structure into noisy per-context maps). compositional_rule: stable -0.0016 (2/4), post-flip -0.0168 (0/4; last_reward post-flip -0.0223 -- memorization liability under shift, echoing K3C's R-table finding in the D channel). Mechanism reading: the |e0| advantage IS per-context base-rate/residual memorization by the D[ctx] tables -- it helps when per-context structure is stable and data-rich (delayed_reward branch), hurts under distribution shift (compositional_rule post-flip) or fragmentation (delayed_multistep stage); on changing_rule the flip-survival rules out pure old-phase contingency memorization and implicates per-context correction of L0's systematic approximation biases on rule-invariant channels. Caveat: the post-flip held-out mixes pre- and post-adaptation trials of the closed-loop reference policy -- not a clean new-regime test. Maturity stays INTEGRATED: no upgrade (thin on changing_rule, dead on 2/4 envs; the |e0| channel is real but narrow). **LOAD-BEARING TEST 2026-10-07 (EXP-AB-K3E, 4 fresh seeds {77101..77104}, preregistered before run; receipts `../../receipts/EXP-AB-K3E.json` + `receipts/EXP-AB-K3E.json`, both hash-verified, G0b determinism PASS):** does the delayed_reward branch-channel |e0| advantage (+0.0986, the program's strongest hierarchy signal) convert into closed-loop control? NO — **PREDICTION-ONLY DECORATION** (bound recorded as NR-B-011). Arm A (intact vs L1-lesioned closed-loop, 30 eps/arm): per-seed gaps +0.0013/+0.0167/+0.1247/+0.0360, seed-mean +0.0447, 4/4 positive — the preregistered gate (seed-mean > +0.05 AND >=3/4) does NOT fire (magnitude bar missed by 0.0053). Both arms dither (shaping-only returns); intact reached the terminal on 2/30 episodes on one seed vs 0/120 lesioned episodes — suggestive but sub-threshold, not significant. Arm B (frozen planners from one K3D-faithful trained model; per-branch predict_reward vs L0 global-map predict_reward_global, new additive knob): per-seed gaps +0.0453/-0.0033/-0.2773/-0.0013, seed-mean -0.0592, 1/4 — does NOT fire; on the two seeds where planners walk the corridor both reach the terminal ~40% of episodes with no systematic per-branch edge, and on seed 77103 the global map wins by +0.28 (per-branch R_ctx tables overfit branch-conditioned reward noise; the correct branch is hidden 50/50, so pooling wins). Mechanism: ArchB's selector never queries predict_next (verified in agent.py _select_action) — the |e0| channel is disconnected from action selection by architecture; the only per-branch path into the selector (R_ctx) cannot carry signal where the branch-contingent outcome is unpredictable. Maturity stays INTEGRATED: the hierarchy's strongest signal is real as prediction, not load-bearing for control. |
| PrecisionEstimator (pi from error stats) | **CAUSAL, REJECTED (both forms)** | C2 KILL (replicated 2 seeds): uniform pi=1 beats estimated pi by 5.8–9.5 return points on changing_rule. C2B (3 seeds): the ONE preregistered shift-aware variant (surprise-triggered window reset, k=4.0) also loses — shift arm 23.97 < estimated 24.10 < uniform 28.47; stationary arm 24.40 < uniform 34.00. Diagnosed: max-channel surprise detector fired 82–110×/run (expected ~1 — cannot separate rule flips from single noisy trials), and post-reset cold-start reintroduces overshoot via tiny-window pi estimates. Do not ship in either form. **C2B 5-seed replication 2026-10-07 (repro5_EXP-AB-C2B): revival fails 0/5 — shift_reset 25.60 < uniform 28.68 (shift), 26.60 < 33.28 (stationary).** **EXP-FP-0008 (2026-10-07, 4 fresh seeds, receipt `../../receipts/EXP-FP-0008.json`): precision-explosion characterization on intact models — ABSENT. The K3C-pilot explosion signature (terminal mean|rerr|_est ≥ 3× uniform AND > 0.3) fires on 0/3 tasks at the ≥3/4-seed bar (pomaze 1/4, delayed_reward 0/4, changing_rule 0/4). Mechanism secondaries: the weight-inflation half operates (b_r/max|w_r| inflate vs uniform on spike tasks) but R_ctx context tables absorb it — estimated piR on terminal trials sits at 3.3–5.4, not pinned at pi_max=20. The C2 pathology family is BOUNDED to lesioned/no-absorber configurations, not a general estimated-precision phenomenon (consistent with C2's FRAGILE 3/5). First run voided pre-interpretation per frozen G0c (rounded-vs-unrounded gate bug; arm verified bit-identical; rerun same seeds). Recorded as NR-B-010.** |
| EpisodicStore (experience_store port + provenance) | **CAUSAL** | M1 (3 seeds): disabling the store selectively degrades pomaze (seed-mean delta +2.200; all 3 seeds +1.83…+2.39) vs delayed_reward (+0.046). Retrieval correction + per-action error prediction are load-bearing on the partial-observability task. NOT decorative. Scope-bounded: shown on pomaze/delayed_reward only, not generalizing. **REPRODUCED 2026-10-07 (repro5_EXP-AB-M1, 5 fresh seeds {75101..75105}): selective deficit 5/5, pomaze deltas +1.71…+2.80 (seed-mean +2.158), delayed_reward +0.03…+0.09 (seed-mean +0.050).** |
| consolidation_cycle race (S-01 vs memory_port, EXP-FP-0005) | **EXECUTED (negative result)** | 2026-10-07, 4 fresh seeds, pomaze corpus, all gates passed (receipts `../../receipts/EXP-FP-0005-S.json`, `../../receipts/EXP-FP-0005-D.json`). Offline replay shows NO measured probe improvement vs no-replay: seed-mean IG P1 (S-01) +0.049, P2 (memory_port) −0.064, U (uniform) +0.323, N 0.0; G1 fails for both prioritized arms (P1-vs-N 2/4, P2-vs-N 1/4 per-seed wins). Uniform replay beats BOTH prioritized arms 4/4 — PE-magnitude prioritization actively hurts (replays noisy/outlier transitions; uniform covers the distribution). Implementation race: P1 beats P2 4/4 (Δ=+0.113 seed-mean IG) — S-01's "promote ids, replay raw" outperforms memory_port's "replay compressed summaries" on this corpus. Phase-4 battery "consolidation CAUSAL" gets a BOUND: mechanism behaves per docs, no offline performance lift measured here. First run declared VOID pre-interpretation per G0c (memory_port merge collapsed ~4549 transitions to ~17–21 entries; raw top-200 batch = 17–21 items); amended to multiplicity replay through consolidated entries, implementation untouched; voided numbers discarded. Negative result recorded in `../../research/negative_results.md`. |
| §30 next-obs / action-consequence predictions | **INTEGRATED** | Logged every tick (JSONL, `experiments_out/EXP-AB-BAR.predictions.jsonl`); consumed by action selection. **Calibration battery EXP-FP-CALIB-01 (2026-10-07, 1950 fresh ticks/domain, preregistered): MISCALIBRATED on all domains** — stated uncertainties fail to beat sequential climatology everywhere (D1 skill −0.009, D2 −0.144 under-confident, D3 −0.398 over-confident, D4 −0.056); corr(sigma,|err|) 0.02–0.17. Confidence outputs are not decision-usable as probabilities; "calibration tracked" was bookkeeping, not calibrated forecasts. Receipt: `../../receipts/EXP-FP-CALIB-01.json`. |
| §30 retrieval-usefulness prediction | **INTEGRATED** | Predicted + measured every tick. Gate implemented and tested (EXP-FP-0006, 2026-10-07, 4 fresh seeds, receipt `../../receipts/EXP-FP-0006.json`): the ûhat sign-gate policy (apply iff uhat > 0) COLLAPSES to never-apply on all 4 seeds — cold-start degeneracy (uhat inits at exactly 0.0; strict > blocks from tick 0; predictor trains on realized benefit=0 and never moves). G ≡ R ≡ no-retrieval → ΔGR=0.000 exactly; preregistered rule fires NEGATIVE, but the rejection is of the gate policy, not of ûhat's ranking ability. **EXP-FP-0007 (2026-10-07, 4 fresh seeds {73501–73504}, receipt `../../receipts/EXP-FP-0007.json`): the non-degenerate shadow-trained-predictor instrument (gate reads a second UsefulnessPredictor trained on counterfactual unconditional benefit; `predictions.py` untouched) is non-degenerate (application rate 0.79–0.80, 4/4 seeds) and the preregistered win rule FIRES — H SUPPORTED: seed-mean ΔGU=+0.3023 (3/4 seeds agree), seed-mean ΔGR=+0.2532 (4/4 agree). The ranking signal carries decision-usable value once the cold-start degeneracy is removed.** Clean-data ranking check replicates on fresh seeds (arm-U logs): corr(uhat, benefit)=+0.405…+0.464 (8 seeds total: +0.40…+0.53), P(benefit>0)≈0.67. First run voided pre-interpretation per frozen G2 (gate compared rounded vs unrounded mean — implementation bug, arm verified bit-identical; numbers discarded, gate fixed, rerun same seeds). Honest bounds: all returns negative (pomaze unsolved by all arms) — the gate adds ~+0.30 on a −3.5 baseline; single env, 4 seeds. CAUSAL label withheld pending replication breadth. **Replication breadth FAILED — EXP-FP-0007R (2026-10-07, 5 fresh seeds {80001–80005}, pomaze/delayed_reward/changing_rule, identical instrument, receipt `../../receipts/EXP-FP-0007R.json`): win rule fires on NONE of the three envs (pomaze ΔGU seed-mean −0.1805 1/5, ΔGR −0.0207 3/5; delayed_reward ΔGU −0.0093 2/5; changing_rule ΔGU −0.4667 2/5, ΔGR +0.2667 3/5 — positive mean but misses ≥4/5). The INTEGRATED promotion is BOUNDED to the 4-seed EXP-FP-0007 pomaze result (9-seed combined seed-mean ≈ +0.09 dGU / +0.13 dGR). The ranking correlate itself still replicates (pomaze corr +0.38…+0.48, 13 seeds total) but the decision value does not — INTEGRATED held, CAUSAL rejected.** Calibration battery D4 (2026-10-07): MISCALIBRATED as a probabilistic claim (observed benefit rate flat ~0.40–0.47 across predicted 0.08–0.64) — ranking is decision-usable, probabilities are not. **EXP-FP-0009 signal-quality characterization (2026-10-07, 5 envs x 5 fresh seeds {81001-81005}, preregistered experiments/preregistration_SIGNAL_QUALITY.json, receipt `../../receipts/EXP-FP-0009.json`, hash-chained, G2 determinism 1e-12 all envs): shadow predictor trained on unconditional-correction streams; per-env seed-mean corr(uhat,benefit): pomaze +0.431, cue_delayed_reward +0.358, delayed_reward +0.266, changing_rule +0.030, compositional_rule +0.024. Signal quality tracks benefit base rate P(benefit>0) (pomaze 0.711 / delayed 0.787 / cue 0.663 vs changing 0.447 / compositional 0.413). Where the signal exists it lives more in MAGNITUDE than SIGN (corr(uhat,|benefit|) 0.497/0.326/0.238 vs corr(uhat,sign) 0.266/0.223/0.163). Carrier features: pomaze = e1_ema (+0.32, w +0.51 aligned - high-error moments benefit most); delayed_reward/cue_delayed_reward = n_nbrs NEGATIVELY (-0.54/-0.60, weights negative aligned - sparse-memory ticks rely on correction; mean_sim also negative but weight sign NOT learned - LMS converged on the n_nbrs channel); changing_rule/compositional_rule = NO feature exceeds |0.08| - regime shifts destroy the feature->benefit mapping, base rate <0.5 drags the bias negative, which explains BOTH the weak correlate AND the low gate rate (0.31-0.35) from one root cause. Gate apply rate at threshold 0: 0.789/0.888/0.771 vs 0.353/0.312. Correction availability ~0.93-0.995 on ALL envs (deterministic correction pass, experiments_out/EXP-FP-0009.avail_correction.json) - availability is not the bottleneck, predictability of benefit is. Methods note: sealed receipt field 'avail_rate' is denominator-buggy (agent._tick resets per episode, values ~14.6-15.0x); corrected values in the correction file above, determinism cross-checked bit-identical. INTEGRATED held, no maturity change (characterization, not a promotion experiment).** |
| ActiveInferenceSelector | **INTEGRATED, label removed PERMANENTLY** | C4 KILL: greedy beat AI on preregistered IG metric (0.0502 > 0.0275). C4B (3 seeds): on the sharper uncertainty-reduction metric (probe-set prediction-error decline), AI 0.5443 < greedy 0.7124 ≈ random 0.5497 — kill confirmed permanently. The IG term's causal contribution is unproven under both metrics. Note: the probe metric rewards concentrated practice (greedy's narrow-deep experience), a caveat for future IG metric design — it does not change the verdict. **C4B 5-seed replication 2026-10-07 (repro5_EXP-AB-C4B): AI 0.4476 < greedy 0.6487 ≈ random 0.5530, kill holds 5/5.** |
| ErrorAffect (error-derived valence/arousal) | **CAUSAL, REJECTED** | K4 KILL: PAD (-0.762) beat error-affect (-0.787) on resource_world. NR-B-002: arousal saturates → permanent over-exploration. Dropped; PAD kept as baseline. |
| PadController (donor-exact baseline) | **CAUSAL** | Won K4. Kept as the cheaper baseline per the kill rule. |
| self/world distinction (EXP-SW-01-B, §9 item 12) | **CAUSAL** | 2026-10-07, 4 fresh seeds {75101–75104}, new self_world v1.0.0 env (hand SELF-caused, ball WORLD-caused, matched change statistics), open-loop learn_transition training (600 transitions), held-out predict_next eval (300 transitions). DISTINCTION DETECTED: seed-mean D_adj = +0.091 (> +0.05 gate), D_adj > 0 on 4/4 seeds (+0.019…+0.174); D_intact stable 4/4 (+0.109…+0.128) — intact mean_abs_e hand ≈ 0.05–0.06 vs ball ≈ 0.16–0.19. D_adj = D_intact − D_frozen on the IDENTICAL held-out stream (paired bias correction; pre-run amendment after the throwaway pilot showed the frozen arm gaps positive from position-marginal sampling noise). Interpretation: the action-conditioned generative model predicts self-caused changes better than world-caused ones — the prediction-error gap is LEARNED (frozen shows only stream noise), i.e. B carries the self/world distinction in its prediction error. Bounds: single env, open-loop training only; per-seed D_adj variance is frozen-noise dominated on 2/4 seeds. Receipt: `../../receipts/EXP-SW-01-B.json` (hash-chained). **EXP-SW-02-B (2026-10-07, closed-loop extension, 4 fresh seeds {75201–75204}, B-only by preregistered design): the distinction does WORK. New additive env self_world_cl v1.0.0 (self_world subclass, reward = −(hand−0.5)²; canonical env untouched). Arms: intact (open-loop trained, then frozen) vs action-shuffle (identical streams/budget/seeds, shuffled action labels — the attribution lesion) vs frozen; greedy one-step planner on model.predict_next drives action selection. ATTRIBUTION ADVANTAGE (PASS): seed-mean R_gap_attrib = return_intact − return_shuffle = +13.59 (> +1.0 gate), positive 3/4 seeds (+41.40 / 0.00 / +9.76 / +3.20); intact hits the predicted limit cycle (−6.0, 4/4) vs frozen −47.4 (4/4). Mechanism diagnostic: intact action term +0.33, shuffle noise −0.048…+0.031 (~10× smaller). Honest caveat: seed 75202's lesion noise carried the correct sign, tracing the intact trajectory exactly (gap 0.0) — the lesion is statistical, not surgical; the preregistered rule allowed one such seed. Online D_intact = +0.117…+0.165 during control (the EXP-SW-01 distinction persists under closed-loop action selection); online D_shuffle ≈ 0. G0c sanity true 4/4. Receipt: `../../receipts/EXP-SW-02-B.json` (hash-chained). CONSCIOUSNESS: UNRESOLVED — a control advantage is a mechanism, not a subject.** |

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
2. Retrieval-usefulness prediction: gate IMPLEMENTED and TESTED (EXP-FP-0006, 2026-10-07) — the sign-gate policy collapsed to never-apply on all 4 seeds (cold-start degeneracy; not self-bootstrapping) and was REJECTED as a policy. **EXP-FP-0007 (2026-10-07): the non-degenerate shadow-trained-predictor instrument is non-degenerate (rate 0.79–0.80) and the preregistered win rule FIRES — H SUPPORTED (seed-mean ΔGU=+0.3023, ΔGR=+0.2532). Gate policy PROMOTED to INTEGRATED** (wired, measured behavioral value vs no-gate and chance-gate; CAUSAL withheld pending replication breadth). The predictor learns a real ranking signal (corr(uhat,benefit)=+0.40…+0.53 across 8 seeds) that is now exploited by a working gate. Bounds: pomaze only, 4 seeds, ~+0.30 on a −3.5 baseline.
3. Hierarchy generalization beyond changing_rule/delayed_reward TESTED 2026-10-07 (K3C): does NOT generalize — lesion gaps vanish/go negative on delayed_multistep and compositional_rule (4/4 seeds each). The hierarchy is a two-env phenomenon; context tables are a liability under distribution shift (post-hoc phase-7/8 diagnostic).
4. Precision weighting rejected in both tested forms (estimated, shift-reset);
   no further precision variants preregistered — the idea is shelved, not
   merely paused. **EXP-FP-0008 (2026-10-07) bounds the C2 pathology family:
   the precision-explosion signature is ABSENT on intact models (0/3 tasks);
   the weight-inflation mechanism operates but R_ctx tables absorb it. The
   pathology needs lesioned/no-absorber configurations — consistent with
   C2's FRAGILE 3/5.**
5. Self/world distinction (EXP-SW-01-B, 2026-10-07): DETECTED on one env
   (self_world) under open-loop training — the prediction-error gap is
   learned and paired-bias-corrected, but multi-env generalization is
   untested. **Closed-loop attribution TESTED 2026-10-07 (EXP-SW-02-B):
   ATTRIBUTION ADVANTAGE (PASS) — the distinction does control work
   (seed-mean R_gap_attrib +13.59, 3/4 seeds positive; intact action term
   +0.33 vs shuffle noise ~10× smaller). One seed's lesion noise aligned
   helpfully (gap 0.0) — the lesion is statistical, not surgical.**
   Architecture A shows NO distinction on the same probe (expected
   negative; no action-conditioned path exists in A).

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

## 2026-10-07 — EXP-FP-0010 pomaze diagnosis: B = EXPLORATION + REPRESENTATION bound + weak credit retention

Cross-architecture pomaze diagnostic (6 fresh seeds 91001–91006; receipt
receipts/EXP-FP-0010-POMAZE-DIAG.ndjson). B-STD: mean −3.58, p_goal 0.133
(22× uniform random's 0.006, still rare). Measured:
- EXPLORATION: IG-driven exploration moves (stuck 0.505 ≈ random) but
  does not systematically search the tree maze; reward is genuinely
  sparse under random play (1/180).
- REPRESENTATION: n_contexts == 1 confirmed (no categorical obs fields
  → single global context; position must ride mu0≈obs). Obs aliasing
  0.667 — the 5-channel position signature (wall bits + beacon, 28 keys
  observed) collides across positions with disagreeing BFS-optimal
  actions in 14/21 multi-position keys. The linear reward head cannot
  resolve position (same function-class lesson as EXP-FP-0002 on D).
- CREDIT (transient, not retained): goal-trial |rerr| declines where
  rewards are frequent (NEAR probes 1.124→0.984; 88 pooled probes
  1.227→0.985) but only 41% of probes are closer to 1.0 when
  re-evaluated at run end (mean drift +0.077 — washed out by the flood
  of −0.01 trials). Dense shaping is NOT learned: mean|rerr| flat across
  B-DENSE episodes; p_goal 0.100 vs 0.133 — no navigation gain from dense
  reward. So B's failure is NOT pure exploration.
- DEMO (5 NEAR then 15 canonical): ZERO transfer (B-DEMO ≡ B-STD:
  −3.62, p_goal 0.133).
- Maturity impact: none of the component levels move (no new causal
  claim); the pomaze bound is now characterized as a compound failure,
  not a single-mechanism one. Follow-up lead: separate head-learning
  from selector-use on the dense variant (does rhat learn the beacon
  gradient while the IG selector ignores it?).

## 2026-10-07 — EXP-FP-0011 aliasing causal test: obs aliasing is NOT the binding constraint (NULL_HOLDS)

Causal follow-up to the EXP-FP-0010 B diagnosis (aliasing 0.667, n_contexts=1,
p_goal 0.133). Intervention: driver-side odometry wrapper
(experiments/odom_wrapper.py, additive — B's core untouched) appending a
coarse relative-position hash (odom_qx/odom_qy, 4x4 cells) computed from the
obs stream + chosen actions only (no goal position, no walls, no absolute
position, no env internals). 6 fresh paired seeds 93001–93006, 15
episodes/arm/seed, preregistered before run
(experiments/preregistration_ALIASING.json); receipt
receipts/EXP-FP-0011-ALIASING.ndjson (35 records, hash-chained, verified).
- H required p_goal to at least double AND mean return +0.5 vs paired
  baseline. Observed: B-BASE mean −3.7270 / p_goal 0.1000; B-AUG mean
  −3.3063 / p_goal 0.1556. Gate_a: 0.1556 ≥ 0.2000? NO. Gate_b: +0.4207 ≥
  0.5? NO. **Verdict: NULL_HOLDS.**
- §40 kill arm B-SHUF (identical channel format, odometry driven by an
  independent seeded random stream — decorrelated from true position):
  mean −3.3058 / p_goal 0.1556, indistinguishable from B-AUG
  (Δ +0.0005 return, 0.0000 p_goal). The small sub-gate lift is fully
  explained by the shuffle arm — it carries zero disambiguation
  information, so the +0.42 is extra-channel/exploration noise, not
  position resolution. Had the gates fired, H would have been rejected
  by the kill arm regardless.
- Post-hoc descriptive: the final B-AUG reward head DOES weight the odom
  channels (seed 93001: w_r odom mean|w| = 0.046 vs native 0.156) — the
  feature is absorbed but not load-bearing. B reaches the goal too rarely
  for any representation to consolidate.
- Control: same augmentation on changing_rule (inert constant channels):
  23.2333 → 22.5889, Δ = −0.6444 ≥ −1.0 → PASS; existing competence intact.
- Frozen gates all pass: G0 canonical files byte-identical pre/post, G1
  tripwire CLEAN, G2 determinism −2.508000 vs −2.508000 MATCH, G3 chain
  verified, G4 nothing pushed.
- Maturity impact: the 0010 REPRESENTATION characterization stands as
  measurement (aliasing 0.667 is real), but causally it is NOT the binding
  constraint — B's pomaze failure remains EXPLORATION-dominant. No
  component level moves. Mechanism lead: representation can only bind once
  reward is found repeatedly — combine disambiguation with a
  systematic-search exploration intervention. Recorded as a negative in
  research/negative_results.md.
