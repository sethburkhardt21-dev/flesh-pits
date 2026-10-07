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
