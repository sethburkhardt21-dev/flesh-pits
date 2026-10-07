# Negative results — Flesh Pits (§40, §52)

Negative results are legitimate success when they eliminate bad hypotheses.
Each entry: experiment ID, what was expected, what happened, what it rules
out. Preregistrations live in experiments/EXPERIMENT_REGISTRY.md; per-run
evidence in receipts/.

## 2026-10-07 — EXP-FP-0006: the ûhat sign gate collapses to never-apply (B gap #2 — CLOSED BY REJECTION OF THE POLICY)
- Expectation: gating the retrieval correction on ûhat (apply iff uhat > 0) would beat unconditional correction on pomaze mean return, and beat random gating at the same rate — proving ûhat carries decision-usable ranking signal.
- Observed (4 fresh seeds {73401..73404}, 15 pomaze episodes/arm/seed, preregistered before the run): the gate NEVER applied the correction on any seed — application rate 0.000 (n_available 2621–2999, applied 0). Mechanism: cold-start degeneracy — uhat initializes at exactly 0.0, the strict `>` blocks from tick 0, the predictor then trains on the realized gated-outcome (benefit=0), and uhat stays at exactly 0.0 forever. The sign-gate policy is not self-bootstrapping. G ≡ R ≡ no-retrieval, so ΔGR = 0.000 exactly on all 4 seeds — the preregistered G-vs-R comparison is vacuous, not a test of ranking. Per-seed ΔGU (gated vs ungated): +0.231/+0.104/−0.325/+0.151 (seed-mean +0.040) — skipping vs applying the correction shows no measurable return difference on pomaze.
- The negative is NOT "ûhat cannot rank": the clean-data ranking check (arm-U PredictionLogs, unconditional corrections, in-sample, n≈2600–3000/seed) gives corr(uhat, benefit) = +0.403…+0.483 across all 4 seeds, P(benefit>0)≈0.67, mean uhat tracking mean benefit. The predictor learns a real ranking signal — this gate policy cannot exploit it. (Consistent with EXP-FP-CALIB-01 D4: the *probabilistic* claim is miscalibrated; the *continuous ranking* has signal.)
- Rules out: "a sign-of-predicted-benefit gate with a cold start is a usable retrieval policy." Exploiting the ranking signal requires a non-degenerate instrument (shadow-trained predictor, warm start, or a non-strict/warmer threshold) — a future experiment, not a retrofit of this one (no post-hoc tuning per the abstention rule).
- The predictor stays EXECUTED; gap #2 is closed by rejection of this policy, not by integration. Receipt: receipts/EXP-FP-0006.json (hash-chained).

## 2026-10-07 — EXP-FP-0001: D does not navigate grid_world
- Expectation: Architecture D's predictor + episodic memory + workspace would
  beat the stateless random baseline by learning to approach the goal.
- Observed: D mean_return=-2.03 vs stateless -2.13 over 30 episodes
  (primary_seed=7). Neither reached the goal once (0/30 positive-return
  episodes each). D's advantage (+0.104) is real but substantively empty.
- Rules out: "D's machinery suffices for goal-directed navigation." The
  greedy-on-predicted-value policy with no exploration does not navigate.

## 2026-10-07 — EXP-FP-0002: D does not learn cue contingencies
- Expectation: D's predictor would learn cue→action mappings and re-learn
  after the rule flip; post-flip return would clearly exceed the no-learning
  baseline.
- Observed: post-flip mean 20.40 vs 19.60 (chance = 20/40). P(correct action)
  per episode: 0.450–0.525, chance throughout.
- Rules out: "a single linear predictor over [obs; onehot(action)] can learn
  cue×action contingencies." The reward is a nonlinear interaction
  (a == rule[cue]); it is not linearly separable in D's features. D's error
  declines only to the linear model's irreducible floor. This is a capacity
  characterization, not a bug — and a concrete bar for Architecture B.

## 2026-10-07 — EXP-FP-0003: D locks into a no-op fixed point
- Expectation: D would at least stumble forward sometimes on delayed_reward.
- Observed: exactly 0.0 return in all 20 episodes. Action trace: [3]*15 every
  episode — pure-greedy + deterministic tie-break + random init picked "stay"
  at t=0 and never tried anything else.
- Rules out: "greedy-on-predicted-value is a non-degenerate policy." Without
  any exploration mechanism, D can lock into no-op loops permanently. Any
  architecture building on D needs an exploration drive or it inherits this
  failure mode.

## 2026-10-07 — EXP-FP-0004: predictor learning adds nothing on pomaze
- Expectation: D (learned predictor) would beat fixed_predictor (frozen) —
  the Architecture B K2 kill experiment in miniature.
- Observed: D -6.00 vs fixed -5.93 over 20 episodes; neither reached the goal
  once. (The symbolic frontier explorer solves 10/10 mazes, so the task is
  solvable — D's policy is the problem, not the task.)
- Rules out: "predictor learning alone improves maze navigation under a greedy
  policy." K2's kill condition is met in miniature: on this task, the learning
  machinery is decorative relative to the frozen predictor.

## 2026-10-07 — build-time: all-history surprise baseline admits nothing
- Expectation: Welford (all-history) mean+std over prediction error would make
  a good surprise gate for episodic admission.
- Observed: the initial learning transient dominates the statistics, so after
  ~10 steps almost nothing new ever exceeds mean+std (store stayed at 1–3
  episodes after 80 steps).
- Rules out: all-history surprise baselines for admission gating. Replaced with
  an EMA (recency-weighted) baseline: surprise is relative to CURRENT
  expectations. (Fixed during build; preserved here as a design negative.)

## 2026-10-07 — build-time: changing_rule rule schedule leaked across episodes
- Expectation: rule seeded by (reset seed, phase) gives a stable phase schedule.
- Observed: the harness uses a fresh episode seed per episode, so the rule
  changed EVERY episode instead of every PHASE_LEN episodes.
- Rules out: seeding phase schedules from per-episode seeds. Fixed: the rule
  schedule is now a fixed deterministic function of phase (identical every
  run); per-episode stochasticity (cue order, noise) still varies by seed.

## 2026-10-07 — build-time: frontier exploration oscillated on unseen walls
- Expectation: BFS-to-nearest-frontier would explore the pomaze systematically.
- Observed: the agent oscillated between two cells forever — cells beyond
  sensed walls were treated as "unknown frontier" because wall adjacency was
  never recorded into the map.
- Rules out: frontier definitions that don't record sensed wall adjacency.
  Fixed: sensed walls are written into the map as known-wall cells; the
  symbolic baseline now solves 10/10 mazes.

## 2026-10-07 — REPRO-K1: B-K1 "learning freeze" verdict does NOT replicate (MAJOR)
- Expectation: the Phase-4 kill EXP-AB-K1 verdict (SURVIVES: learn 26.2 >
  frozen 21.6, +4.6) would hold on fresh seeds.
- Observed: on 5 fresh seeds (72001–72005) the FROZEN agent beats the
  learning agent on ALL 5 (deltas −0.8, −4.0, −4.0, −4.2, −10.2). Kill
  condition met 5/5 — and reversed: freezing IMPROVES post-flip return.
- Driver faithfulness verified: original config (primary=101, agent=11)
  reruns to exactly +4.60. Decomposition: changing ONLY the agent init seed
  (101/72001) flips to −0.20; changing ONLY the env stream (72001/11)
  flips to −5.60. The +4.6 was a lucky draw on both.
- Rules out: "learning is load-bearing for control on changing_rule." The
  frozen agent's random-init weights + mu0 state tracking do most of the
  work. Receipt: prototypes/architecture-b/receipts/repro_EXP-AB-K1.json.

## 2026-10-07 — REPRO-K3: B-K3 "hierarchy lesion" verdict does NOT replicate (MAJOR)
- Expectation: EXP-AB-K3 verdict (SURVIVES, WEAK: structured delta +0.0032)
  would hold on fresh seeds.
- Observed: on 5 fresh seeds (72201–72205) the structured lesion delta is
  NEGATIVE on all 5 (−0.008…−0.028) — disabling L1 consistently HELPS L0
  prediction. White-noise deltas stay small/mixed as preregistered.
- Driver faithfulness verified: original seed 103 reruns to exactly
  +0.0032/−0.0012. The original +0.0032 was a single-seed artifact; the
  consistent direction across 5 fresh seeds is systematic, not noise.
- Rules out: the original K3 |e0| probe as an instrument — seed-fragile and
  diluted by unpredictable channels. (Concurrent K3B worker's longer-horizon
  reward-channel probe is a separate instrument with separate evidence;
  this entry concerns the ORIGINAL probe only.)
- Receipt: prototypes/architecture-b/receipts/repro_EXP-AB-K3.json.

## 2026-10-07 — REPRO-K5: B-K5 "shuffle" verdict does NOT replicate
- Expectation: EXP-AB-K5 verdict (SURVIVES, WEAK: shuffled-trained error
  0.8606 > ordered-trained 0.8575, delta +0.0030) would hold on fresh seeds.
- Observed: on 5 fresh seeds (72301–72305) shuffled-trained is AS GOOD OR
  BETTER on all 5 (deltas −0.0002…−0.0457). Kill condition met 5/5.
- Driver faithfulness verified: exact original config reruns to +0.0030.
- Rules out (per the preregistered kill condition): the temporal-prediction
  claim — the model does not demonstrably learn the temporally-tracked
  rule vs bag-of-features marginals on this probe. Claim demoted.
- Receipt: prototypes/architecture-b/receipts/repro_EXP-AB-K5.json.

## 2026-10-07 — REPRO-C1: B-C1 "error decline" holds only 3/5 seeds
- Expectation: EXP-AB-C1 verdict (HOLDS: |e0| declines first→last quintile
  on delayed_reward and grid_world) would hold on fresh seeds.
- Observed: 3/5 hold. Flips are small-magnitude: seed 72403 grid_world
  decline −0.0075 (delayed_reward +0.2179 on the same seed); seed 72405
  delayed_reward decline −0.0235 (grid_world +0.2157 on the same seed).
  The failure mode is per-env, not systematic — the within-run comparison
  is noisy at the boundary.
- Verdict: MIXED. The claim is directionally right but fragile; do not cite
  C1 as established without the 3/5 qualifier.
- Receipt: prototypes/architecture-b/receipts/repro_EXP-AB-C1.json.

## 2026-10-07 — REPRO-C2: B-C2 "precision ablation" kill is seed-fragile (3/5)
- Expectation: EXP-AB-C2 kill verdict (uniform pi=1 beats estimated pi)
  would replicate on fresh seeds.
- Observed: uniform wins on 3/5 seeds (margins −2.5, −5.4, −9.9); estimated
  precision wins on 2/5 (margins +0.8, +3.5). The kill direction flips.
- The engineering rejection stands (uniform wins more often and by larger
  margins; concurrent C2B worker independently rejects the shift-aware
  variant), but the original single-seed kill margin is seed-dependent —
  cite as KILL, FRAGILE, not as a clean kill.
- Receipt: prototypes/architecture-b/receipts/repro_EXP-AB-C2.json.

## 2026-10-07 — REPRO cross-cutting pattern (for Phase-4 follow-up)
- The replication failures cluster exactly where L1 / precision / learning
  interact with distribution shift: K1 (freeze helps), K3 (lesion helps),
  C2 (uniform wins). Pure prediction learning (B-K2: learned predictor
  beats fixed 5/5, gap 0.405–0.426, rock-stable) is robust.
- Coherent reading: the predictor LEARNS, but the L1-context and precision
  machinery does not robustly convert learning into better decisions under
  shift. All of Architecture A (K1–K4) reproduced 5/5 — the fragility is
  specific to B's adaptive machinery, not the reproduction method.

## 2026-10-07 — EXP-FP-CALIB-01: B's §30 uncertainties are miscalibrated on all four domains
- Expectation: B's §30 stated uncertainties (oconf/ounc, rconf/runc, uconf/uunc) are calibrated probabilistic claims — Brier score would beat sequential climatology on all four domains with reliability near the diagonal.
- Observed: on 1950 fresh closed-loop ticks per domain (5 fresh seeds, changing_rule, preregistered thresholds from 2340 published ticks, no post-run tuning), B fails to beat climatology EVERYWHERE: D1 next_obs 0.2532 vs 0.2508 (skill −0.009, MCE 0.909); D2 action_consequence 0.2866 vs 0.2505 (skill −0.144, under-confident S=−0.19); D3 competence/failure 0.1025 vs 0.0734 (skill −0.398, over-confident S=+0.17, failure over-predicted ~3× in the main bin); D4 retrieval_usefulness 0.2610 vs 0.2472 (skill −0.056, observed benefit rate flat across the predicted range).
- Rules out: "B's confidence outputs are decision-usable as probabilities." Root cause: stated uncertainty is essentially uncoupled from actual error magnitude (corr(sigma,|err|) = 0.02 D1/D3, 0.17 D2, 0.05 D4). The existing CalibrationTracker bin/slope summaries remain descriptive only; the "calibration tracked" maturity label described bookkeeping, not calibrated forecasts. Receipt: receipts/EXP-FP-CALIB-01.json (hash-chained); full table benchmarks/calibration_battery_results.json.

## 2026-10-07 — EXP-FP-0005: the consolidation race — offline replay shows no lift; prioritization hurts (NEGATIVE)
- Expectation (H1, preregistered): offline prioritized consolidation replay (S-01 promotion list or memory_port top-k relevance batch) would causally improve held-out probe prediction error vs no replay, and beat uniform replay at equal budget.
- Observed (4 fresh seeds, all gates passed, receipt receipts/EXP-FP-0005-S.json): seed-mean IG_probe — P1 (S-01) +0.049, P2 (memory_port) −0.064, U (uniform) +0.323, N 0.0. G1: P1-vs-N 2/4 per-seed wins, P2-vs-N 1/4 — neither beats no-replay under the ≥3/4 rule. G2: uniform beats BOTH prioritized arms 4/4. G3 (implementation race): P1 beats P2 4/4 (Δ=+0.113 seed-mean IG) — S-01's "promote ids, replay raw" outperforms memory_port's "replay compressed summaries."
- Rules out: "offline prioritized replay (by prediction-error magnitude) improves later model performance on this corpus." The Phase-4 battery "consolidation CAUSAL" claim gets a BOUND: the consolidation mechanism behaves per docs (mechanism CAUSAL, per brothel EXP-MEMORY-001), but shows no measured offline performance lift here; prioritization by PE magnitude is actively worse than uniform replay (likely replays noisy/outlier transitions while uniform covers the distribution).
- Mechanism note: both implementations merge ~4600 pomaze transitions into ~17–21 groups at merge_threshold 0.9 — pomaze obs vectors are highly mutually similar; the merge is working as designed, the corpus is not near-duplicate-heavy in a semantic sense.
- Design history: the first run was declared VOID pre-interpretation per gate G0c (memory_port's raw top-200 batch yielded only 17–21 items); amended to multiplicity replay through consolidated entries (implementation code and parameters untouched); rerun on the same seeds; voided numbers discarded, never interpreted.

## 2026-10-07 — repro5_EXP-AB-K3B: the hierarchy's reward-channel signal reverses on fresh seeds (K3B weakened, not killed)
- Expectation: the K3B verdict (SURVIVES, STRENGTHENED) would replicate on 5 fresh seeds — arm-A reward-channel lesion gap staying positive at ~19× K3's +0.0032 scale (+0.0602, 3/3 seeds).
- Observed (5 fresh seeds {75401..75405}, identical instrument, preregistered before the run): the preregistered gate fires REPRODUCES at exactly 4/5 seeds — per-seed "grown" (delta_e0 > 0.0032 OR delta_rerr > 0.0032 on arm A) holds on 75401/75403/75404/75405, fails on 75402. BUT the carrying channel reversed: arm-A |rerr| gap = −0.0319 seed-mean, negative on 3/5 seeds (75402 −0.0305, 75403 −0.0855, 75404 −0.2753 — lesion HELPS reward prediction there). The replication's signal rides the |e0| channel instead (+0.0079 seed-mean, positive 4/5), at only ~2.5× K3's scale. Arm B: |e0| +0.0356 (reproduces the +0.0437 direction); terminal |rerr| −0.0224 (≈0, as preregistered).
- The negative: the +0.0602 reward-channel gap — the headline evidence that the hierarchy "earns its keep on longer horizons" — was seed-fragile. It does not survive fresh seeds. The hierarchy effect on longer horizons is real in a thin sense (4/5 seeds show some above-K3-scale growth on at least one channel) but its locus (reward vs dynamics channel) and magnitude are not stable.
- Rules out: "K3B's reward-channel gap is a robust property of the hierarchy on longer horizons." The scope bound tightens: the hierarchy's longer-horizon contribution is, at best, a weak dynamics-channel effect (~2.5× K3 scale), and the two-env phenomenon stands (K3C).
- The preregistered verdict is REPRODUCES (frozen gate, 4/5) — this entry records the substantive weakening, not a verdict change. Receipts: receipts/repro5_EXP-AB-K3B.json + prototypes/architecture-b/receipts/repro5_EXP-AB-K3B.json (hash-chained).

## 2026-10-07 — EXP-SW-01-A: Architecture A carries no self/world distinction (§9 item 12)
- Expectation (preregistered as the null): no internal variable of A would systematically distinguish self-caused from world-caused observation changes under matched statistics — A's specialists, arbitrator, and ignition have no action-conditioned path.
- Observed (4 fresh seeds {75101..75104}, self_world v1.0.0, closed-loop WorkspaceTick, preregistered before the run): seed-mean Delta_bid = +0.00088 (rule needed |Delta| > 0.05), Delta_ign = 0.000 — nothing ever ignited on either channel (habituated bids sit below the ignition threshold); final gains floored symmetric at 0.01/0.01. The preregistered DISTINCTION rule did not fire on any seed.
- Mechanism: the z-score bid habituates per channel to 0.5 regardless of cause, erasing even unmatched change statistics; ignition's bistable gate then admits nothing. A has no efference copy — nothing in the tick conditions perception on the agent's own action. The measurement chain is not vacuous: a unit test proves the specialist→bid chain discriminates a large unilateral deviation (test_chain_can_discriminate).
- Rules out: "A's workspace machinery implicitly tracks which observation changes it caused." Any future self-model for A needs an explicit action-conditioned path; the current self_model_update consumer (broadcast counts, ignition EMA, last channel) cannot represent causal attribution.
- Receipt: receipts/EXP-SW-01-A.json (hash-chained).

## 2026-10-07 — EXP-FP-0008: precision explosion does NOT systematize on intact models (C2 pathology family BOUNDED)

- Expectation (preregistered): the estimated-precision reward head would systematically misbehave on rare high-reward events — explosion signature (held-out terminal mean|rerr|_est ≥ 3× uniform AND > 0.3) on ≥3/4 seeds on at least 2 of 3 tasks (pomaze, delayed_reward; changing_rule as dense-reward negative control). Background: the EXP-AB-K3C task-1 pilot saw the lesioned model's reward head explode on rare terminal +1.0 spikes (b_r=0.31, max|w_r|=0.73, rhat=1.27 on a 0.02 tick; 0.83 vs 0.10 vs uniform) — a second independent sighting of the C2 precision pathology family.
- Observed (4 fresh seeds {74301..74304}, intact ArchB, open-loop K3B-faithful instrument): **ABSENT** — 0/3 tasks reach the ≥3/4 bar (pomaze 1/4, delayed_reward 0/4, changing_rule 0/4). The weight-inflation half of the pathology DOES operate (b_r/max|w_r| inflate vs uniform on spike tasks), but the R_ctx context tables absorb it — estimated piR on terminal trials sits at 3.3–5.4, not pinned at pi_max=20 — so held-out terminal |rerr| does not systematically explode.
- Rules out: "estimated precision misbehaves on rare spikes, generally." The C2 pathology family is BOUNDED to lesioned/no-absorber configurations (consistent with C2's FRAGILE 3/5 kill). Characterization, not a kill — recorded as arch-b NR-B-010.
- Methods note: first run voided pre-interpretation per frozen G0c (rounded-vs-unrounded gate bug; arm verified bit-identical; numbers discarded, gate fixed, rerun same seeds).
- Receipt: receipts/EXP-FP-0008.json (hash-chained; full per-task/per-seed detail in summary).

## 2026-10-07 — addendum to EXP-FP-0006: the "future experiment" ran (EXP-FP-0007, H SUPPORTED)

- The 0006 entry above closed gap #2 "by rejection of the policy" and named the needed future experiment: a non-degenerate instrument for the ranking signal. That experiment is EXP-FP-0007 (shadow-trained predictor — the gate reads a second UsefulnessPredictor trained on counterfactual unconditional benefit; `predictions.py` untouched).
- Observed (4 fresh seeds {73501..73504}, 15 pomaze episodes/arm/seed, preregistered): the shadow instrument is non-degenerate (application rate 0.79–0.80, 4/4 seeds) and the preregistered win rule FIRES — seed-mean ΔGU=+0.3023 (3/4 seeds agree), seed-mean ΔGR=+0.2532 (4/4 agree). The ranking signal (corr +0.40…+0.53 across 8 seeds) carries decision-usable value once the cold-start degeneracy is removed.
- Status change: gap #2's gate policy is PROMOTED to INTEGRATED (wired, measured behavioral value vs no-gate and chance-gate). Bounds: pomaze only, 4 seeds, ~+0.30 on a −3.5 baseline; CAUSAL withheld pending replication breadth. The 0006 rejection stands as the correct verdict on the sign-gate *policy*; it was never a verdict on the ranking signal.
- First 0007 run voided pre-interpretation per frozen G2 (same rounded-vs-unrounded gate-bug class; arm verified bit-identical; rerun same seeds).
- Receipt: receipts/EXP-FP-0007.json (hash-chained).

## 2026-10-07 — EXP-FP-0007R: shadow-gate advantage does NOT replicate (replication breadth FAILED)

- Expectation (preregistered): the EXP-FP-0007 result (shadow-ûhat gate beats ungated AND random-gated, INTEGRATED) replicates on pomaze at a 5-fresh-seed bar and generalizes to ≥1 of two new envs. Win rule per env: seed-mean ΔGU>0 AND ΔGR>0 with ≥4/5 seeds agreeing on both. Promotion bar: pomaze win fires AND ≥1 new env fires → CAUSAL; otherwise no promotion.
- Observed (5 fresh seeds {80001..80005}, zero overlap with any prior lab seed set, 15 episodes/arm/seed, identical instrument reused verbatim via import): **NEGATIVE — bar envs: none.** pomaze: seed-mean ΔGU=−0.1805 (1/5 agree), ΔGR=−0.0207 (3/5 agree). delayed_reward: ΔGU=−0.0093 (2/5), ΔGR=−0.0032 (3/5) — deltas essentially zero, no retrieval leverage in the env. changing_rule: ΔGU=−0.4667 (2/5), ΔGR=+0.2667 (3/5) — the only positive seed-mean, but 3/5 agreement misses the ≥4/5 bar.
- Rules out: "the EXP-FP-0007 advantage replicates robustly / generalizes." The EXP-FP-0007 INTEGRATED promotion is now BOUNDED to its 4-seed pomaze result (5-against-4 across seeds now; combined seed-mean over 9 seeds is ≈+0.09 for dGU, ≈+0.13 for dGR — both below any promotion-worthy magnitude).
- What DID replicate (descriptive, not the bar): the *ranking signal itself* is stable — pomaze corr(ûhat, benefit) = +0.38…+0.48 on shadow data and +0.40…+0.44 on clean U-arm data (now 13 seeds total). On changing_rule the ranking signal is genuinely weaker (+0.09…+0.14 shadow, +0.06…+0.25 clean) and the gate applies far less often (rate 0.27–0.42 vs 0.71–0.82 on pomaze) — env-dependent signal quality, a mechanism lead for gap #2.
- No promotion: the MATURITY retrieval row stays INTEGRATED, annotated with the failed replication. INTEGRATED means "wired, non-degenerate, one measured win" — the single win is no longer claimed as robust.
- Methods note: all frozen gates PASS (G0 5/5 discriminating all envs; G1 tripwire CLEAN; G2 determinism recompute bit-identical 1e-12 all envs; G3 hash-chained receipt, own hash verifies, chained to EXP-SW-02-B). One executor hiccup: the run crashed at receipt-write (config_hash function object passed instead of its output — executor bug, no data interpretation before the fix; deterministic rerun reproduced every arm number bit-identically, receipt written on the clean run). First launch was also interrupted by a runtime restart mid-run (partial logs discarded, deterministic rerun from the preregistered seeds).
- Receipt: receipts/EXP-FP-0007R.json (hash-chained) + prototypes/architecture-b/receipts/EXP-FP-0007R.json (pre-chain detail). Preregistration: experiments/preregistration_EXP-FP-0007R.json (sealed before run). Executor: prototypes/architecture-b/exp_retrieval_gate_shadow_repl.py.

## 2026-10-07 — EXP-AB-K3E: the branch-channel |e0| advantage does NOT convert into closed-loop control (bound, arch-b NR-B-011)

- Expectation (preregistered): the K3D branch-channel |e0| advantage on delayed_reward (+0.0986 seed-mean, 4/4 — the program's strongest hierarchy signal) would translate into closed-loop control advantage: intact ArchB beats L1-lesioned ArchB (Arm A), and a frozen per-branch planner beats a global-map planner (Arm B, sharper mechanistic arm). Gate: seed-mean return gap > +0.05 AND positive ≥3/4 seeds (4 fresh seeds {77101..77104}).
- Observed: **PREDICTION-ONLY DECORATION — neither gate fires.** Arm A: +0.0013/+0.0167/+0.1247/+0.0360 per seed, seed-mean +0.0447 (bar missed by 0.0053), 4/4 positive; both arms dither, returns shaping-only. Arm B: +0.0453/−0.0033/−0.2773/−0.0013, seed-mean −0.0592, 1/4; where planners walk the corridor the two tie, and the global map wins one seed by +0.28 (per-branch R_ctx tables overfit branch-conditioned noise).
- Rules out: "the hierarchy's strongest signal is load-bearing for control on delayed_reward." Mechanism: ArchB's selector never queries predict_next (verified in code) — the |e0| channel is decoration by architecture; the only per-branch path into action selection (R_ctx) cannot carry signal where the branch-contingent outcome is unpredictable (hidden correct branch).
- Receipt: receipts/EXP-AB-K3E.json (hash-chained). MATURITY.md hierarchy row extended; maturity stays INTEGRATED.

## 2026-10-07 — EXP-FP-0010: POMAZE cross-architecture failure diagnosis (DIAGNOSTIC — the failure classes, with numbers)

- Context: pomaze v1.0.0 is the lab's hardest env — no architecture has ever posted a positive mean return (battery: A −3.95, B −3.62, C −5.80). EXP-FP-0004 showed predictor learning adds nothing under a greedy policy; the symbolic frontier explorer solves 10/10 (task solvable, policy is the problem). This diagnostic (preregistered: experiments/preregistration_POMAZE_DIAG.json, sealed before run; 6 fresh seeds 91001–91006; 14 arms) instruments per-episode telemetry for each arch and runs discriminating probe arms: RANDOM, SYMBOLIC, {A,B,C}×{STD,DENSE,NEAR}, {A,B,C}-DEMO (5 near-goal episodes then 15 canonical). DENSE = additive +0.02×beacon shaping (experiments/envs/pomaze_dense.py); NEAR = additive near-goal start, dist≤4 (experiments/envs/pomaze_near.py); canonical pomaze.py byte-identical (sha256 in receipt, G0).
- Reward facts (measured): the ONLY positive reward is the +1.0 goal cell; everything else is −0.01/step, −0.02/bump. Under uniform random play the goal is hit in 1/180 episodes (p=0.006) — reward is genuinely sparse (tree-maze random walk; expected hitting time ≫ 200 steps). Symbolic: 180/180, mean +0.61, 39 steps mean.
- **A — REPRESENTATION (primary).** State-blind by construction: stimuli ⊥ obs (mean |corr(stimulus, beacon)| = 0.046 across 360 channel-episodes). NR-A-006 decay confirmed on pomaze: gains at the 0.01 floor in 98.9% of channel-episodes. Behavior ≈ random walk (stuck 0.535 vs 0.540; return −4.05 vs −4.15; p_goal 0.044 vs 0.006). DENSE: no navigation gain (p_goal 0.033 vs 0.044) — the +1.64 return lift is mechanical shaping collected while wandering, not navigation. NEAR (p_goal 0.744): gains leave the floor (0.789 vs 0.989) and goal rate rises 0.711→0.778 second-half — the delta rule responds — but DEMO shows ZERO transfer to canonical mazes (p_goal 0.056 vs 0.044; gains re-floor at 0.986). A bandit cannot retain maze-navigation policy across maze seeds: credit assignment works locally, representation forbids generalization.
- **B — EXPLORATION-dominant, but NOT pure exploration: + REPRESENTATION bound + weak credit retention.** p_goal 0.133 on canonical (22× random, still rare); IG exploration moves (stuck 0.505) but doesn't systematically search. n_contexts = 1 confirmed (no categorical obs fields → single global context). Obs aliasing 0.667: the 5-channel position signature collides across 21/28 observed keys, 14 with disagreeing BFS-optimal actions — the linear reward head cannot resolve position (same function-class lesson as EXP-FP-0002). Goal trials register TRANSIENTLY (|rerr| on 88 probes: first-half 1.227 → second-half 0.985; NEAR-only 1.124→0.984) but do NOT retain: only 41% of probes are closer to 1.0 when re-evaluated at run end (mean drift +0.077 — washed out by the flood of −0.01 trials). Dense shaping is NOT learned: mean|rerr| flat across B-DENSE episodes and p_goal 0.100 vs 0.133 — no navigation gain from dense reward. DEMO: zero transfer (B-DEMO ≡ B-STD: −3.62, p_goal 0.133).
- **C — CREDIT ASSIGNMENT via the NR-A-006 decay through the predictor-stimulus path (CORRECTS the build-and-beat autopsy).** The autopsy's "flat predictor outputs" and "tie-break degeneracy" are DISPROVEN by direct measurement: per-episode stimulus std 0.51–1.55 (mean 0.97 — not flat); arbitration margin <1e-9 on 0.001 of ticks (not tie-break; margins 0.001–0.006). What holds: gains decay to floor (0.997 on STD) under sparse punishment. True mechanism: with gains floored, arbitration = argmax of habituated raw bids, and raw bids respond only to stimulus CHANGES (running z-score); a wall-bumping loop yields static obs → static stimuli → frozen winner → a self-reinforcing bump attractor (stuck 0.972; return −5.88 WORSE than random −4.15). The predictor's stimulus variation (largely memory-error bonus/belief dynamics, not goal gradient: stim⊥beacon mean |r| = 0.053) cannot steer the policy because the gain loop floors. DENSE: no navigation gain (p_goal 0.011 = STD; +1.51 lift mechanical). NEAR (p_goal 0.556): gains partially leave floor (0.769), stuck 0.71→0.56 second-half, p_goal 0.467→0.644 — weak response. DEMO: zero transfer (p_goal 0.000, −5.92).
- Rules out: (1) "all three failures are the same exploration failure" — the null is rejected: A≈random-walk, B=22×random with transient registration but no retention, C=self-pinning bump attractor worse than random; probe responses differ sharply. (2) "dense reward fixes pomaze" — no arch gains navigation from the dense variant (all p_goal ≈ STD; all return lifts are the shaping term itself). (3) "demonstrated reward transfers" — 5 near-goal episodes of frequent +1.0 leave NO usable trace on canonical mazes for any arch (DEMO ≡ STD on all three). (4) The build-and-beat "flat predictor / tie-break" mechanism story for C — replaced by the measured bump-attractor mechanism above.
- Honest methods notes: two preregistered discriminator rules need qualification. D1's max-|corr| rule (needed <0.10, got 0.222) was over-strict — the mean (0.046) is the informative statistic and confirms state-blindness; chance excursions over 360 correlations explain the max. The dense-lift rules (D3/D6/D11, fired at +1.64/+1.65/+1.51) did not subtract the shaping term — p_goal (shaping-invariant) shows no navigation gain in any arch, which is the behaviorally honest reading; the rules fired on the letter, the diagnosis uses p_goal. D4/D10's pre/post-first-goal return gaps (+2.95/+4.03) are confounded by within-episode goal hits; the cleaner first-half/second-half p_goal cut shows only modest rises (A 0.711→0.778, B 0.689→0.800, C 0.467→0.644).
- Receipt: receipts/EXP-FP-0010-POMAZE-DIAG.ndjson (92 records, hash chain verified). Preregistration: experiments/preregistration_POMAZE_DIAG.json. Driver: experiments/pomaze_diag.py (additive). Variants: experiments/envs/pomaze_dense.py, experiments/envs/pomaze_near.py (additive; pomaze.py untouched). Frozen gates: G0 additivity PASS (sha256 recorded), G1 tripwire CLEAN (variants + driver, pre-run and after the two bug fixes), G2 determinism MATCH 1e-9 (B-STD 91001 recompute), G3 chain verified, G4 nothing pushed. Two executor bugs fixed before any data was recorded (a _mean generator bug and a DEMO KeyError; no partial receipts existed).
