# EXPERIMENT REGISTRY — Flesh Pits (§38)

Append-only log. Every serious experiment records: experiment ID, hypothesis,
null hypothesis, preregistered metric, baseline, conditions, seed, config hash,
result, interpretation, limitations. No retrospective metric selection: the
metric is frozen BEFORE the run, in the PREREGISTERED block. Results are
appended beneath, never edited into the preregistration.

## Index

| ID | env | agents | preregistered metric | status |
|---|---|---|---|---|
| EXP-FP-0001 | grid_world | arch_d vs stateless | mean_return, 30 eps | PREREGISTERED |
| EXP-FP-0002 | changing_rule | arch_d vs persistent_no_learning | mean_return eps 5–9 (post-flip), 10 eps | PREREGISTERED |
| EXP-FP-0003 | delayed_reward | symbolic vs arch_d | mean_return, 20 eps | PREREGISTERED |
| EXP-FP-0004 | pomaze | arch_d vs fixed_predictor | mean_return, 20 eps | PREREGISTERED |
| EXP-FP-0005 | pomaze | S-01 vs memory_port prioritized vs uniform vs no replay | IG_probe (offline replay improvement), 4 seeds | PREREGISTERED |
| EXP-FP-0007 | pomaze | ungated vs shadow-ûhat-gated vs random-gated | mean_return, 15 eps; ΔGU, ΔGR; win: seed-mean>0 both, ≥3/4 agree | PREREGISTERED |
| EXP-FP-0008 | pomaze/delayed_reward/changing_rule | estimated vs uniform precision heads | terminal-spike mean\|rerr\| ratio ≥3× AND est>0.3; ≥3/4 seeds | PREREGISTERED |
| EXP-AB-K3B | changing_rule + delayed_reward | arch_b intact vs lesion_l1 | lesion gap (lesioned − intact) mean|e0|, mean|rerr|, 3 seeds | COMPLETE (result below) |
| EXP-AB-K3C | delayed_multistep + compositional_rule | arch_b intact vs lesion_l1 | lesion gap D_e0 (task 1), D_rerr (task 2), 4 seeds | COMPLETE (result below) |
| EXP-AB-5SEED | pomaze/changing_rule/delayed_reward | Phase-4 replication (M1, C2B, C4B, K3B) | original per-seed decision rules, ≥4/5 seeds | COMPLETE (result below) |

---

## EXP-FP-0001 — D vs stateless on grid_world

**PREREGISTERED (2026-10-07, before run)**
- hypothesis: Architecture D (predictor + episodic memory + workspace + greedy)
  achieves higher mean episodic return than the stateless random baseline on
  grid_world.
- null: D's mean return ≤ stateless mean return (D's machinery adds nothing
  over random action on this navigation task).
- preregistered metric: mean_return over 30 episodes (from receipt summary).
- baseline: stateless.
- conditions: grid_world v1.0.0, arch_d v1.0.0, 30 episodes, default max_steps,
  contract v1.0.0.
- primary_seed: 7.
- config hash: recorded in receipt at run time.

## EXP-FP-0002 — adaptation after rule flip on changing_rule

**PREREGISTERED (2026-10-07, before run)**
- hypothesis: After the first rule flip (episode 5), D re-learns the new
  contingencies faster than persistent_no_learning (which cannot adapt),
  so D's post-flip mean return exceeds the no-learning baseline's.
- null: no difference in post-flip mean return (D's predictor does not
  usefully track contingency change).
- preregistered metric: mean return over episodes 5–9 (post-first-flip),
  computed from per-episode receipt data; 10 episodes total.
- baseline: persistent_no_learning.
- conditions: changing_rule v1.0.0 (PHASE_LEN=5), arch_d v1.0.0, 10 episodes.
- primary_seed: 11.
- config hash: recorded in receipt at run time.

## EXP-FP-0003 — credit assignment on delayed_reward

**PREREGISTERED (2026-10-07, before run)**
- hypothesis: The deterministic symbolic baseline (fixed branch_a + forward)
  matches or beats Architecture D on delayed_reward, because D's episodic
  store records immediate transition rewards and has no multi-step return
  mechanism — the +1.0 is delivered 10 steps after the causing decision.
- null: D's mean return > symbolic's (D's retrieval somehow bridges the delay).
- preregistered metric: mean_return over 20 episodes.
- baseline: arch_d (the roles reverse here: symbolic is the strong condition).
- conditions: delayed_reward v1.0.0 (LENGTH=10), 20 episodes.
- primary_seed: 13.
- config hash: recorded in receipt at run time.

## EXP-FP-0004 — predictor learning on pomaze

**PREREGISTERED (2026-10-07, before run)**
- hypothesis: D (learned predictor) beats fixed_predictor (frozen random
  predictor) on pomaze — isolating the value of predictor learning, the
  Architecture B kill experiment K2 in miniature.
- null: no difference (predictor learning contributes nothing here).
- preregistered metric: mean_return over 20 episodes.
- baseline: fixed_predictor.
- conditions: pomaze v1.0.0, arch_d v1.0.0, 20 episodes.
- primary_seed: 17.
- config hash: recorded in receipt at run time.

---

## RESULTS (appended after runs; preregistrations above untouched)

### EXP-FP-0001 — RESULT (2026-10-07)
- D: mean_return=-2.0300, stdev=0.0526, min=-2.20, max=-2.00, n=30,
  config_hash=4b3c4072a2de, receipt receipts/EXP-FP-0001-D.json
- stateless: mean_return=-2.1340, stdev=1.1046, min=-4.00, max=-0.38, n=30,
  config_hash=1fb532dcabcb, receipt receipts/EXP-FP-0001-S.json
- Metric (mean_return): D (-2.03) > stateless (-2.13) by +0.104.
- interpretation: The hypothesis is TECHNICALLY true but substantively empty:
  neither agent reached the goal in any of 30 episodes (0 positive-return
  episodes each). D is a consistent wanderer (stdev 0.05: full 100-step
  episodes every time); stateless is variable but equally goalless. D's
  machinery adds ~nothing over random action on navigation.
- limitations: single primary_seed (7); 30 episodes; greedy policy with no
  exploration in D.

### EXP-FP-0002 — RESULT (2026-10-07)
- D: overall mean=20.60; pre-flip (eps 0–4) mean=20.80; post-flip (eps 5–9)
  mean=20.40. config_hash=6b14f6eae469, receipt receipts/EXP-FP-0002-D.json
- persistent_no_learning: overall=19.40; pre=19.20; post=19.60.
  config_hash=4ca4f121ac73, receipt receipts/EXP-FP-0002-S.json
- Preregistered metric (post-flip mean_return): D 20.40 vs 19.60 (+0.8).
- interpretation: NULL EFFECTIVELY HOLDS. Both agents sit at chance (20/40).
  Follow-up measurement: D's P(correct action) per episode over 6 episodes =
  0.450, 0.425, 0.475, 0.500, 0.525, 0.525 — chance throughout. Root cause:
  the reward is a cue×action interaction (a==rule[cue]), which is NOT linearly
  separable in D's [obs; onehot(action)] features. D's single linear predictor
  cannot represent the contingency, so its prediction error declines only to
  the linear model's irreducible floor. This is a precise capacity
  characterization, not a bug: Architecture B's nonlinear/hierarchical
  predictor has a concrete bar to beat here.
- limitations: single primary_seed (11); 10 episodes; PHASE_LEN=5.

### EXP-FP-0003 — RESULT (2026-10-07)
- symbolic: mean_return=0.4800, stdev=0.4583 (bimodal: 1.18 when branch_a
  correct, 0.18 otherwise — fixed branch_a + forward). config_hash=e0167c52b3cd,
  receipt receipts/EXP-FP-0003-S.json
- D: mean_return=0.0000, stdev=0.0000 — exactly 0.0 in all 20 episodes.
  config_hash=d16e380ef0c1, receipt receipts/EXP-FP-0003-D.json
- interpretation: HYPOTHESIS CONFIRMED (symbolic 0.48 > D 0.00). D's failure
  mode is degenerate, not merely weak: action-trace inspection shows D
  repeating action 3 (stay) all 15 steps, every episode. Pure-greedy with
  deterministic tie-breaking and no exploration locks into a no-op fixed
  point at t=0 (random-init argmax picks stay; it is never contradicted
  because it is never tried against alternatives). D has no exploration
  mechanism by design ("greedy on predicted value"); on tasks with
  deceptive no-op actions it scores exactly zero. Documented as a D
  limitation; a D+exploration variant is a future experiment, not smuggled
  into this baseline.
- limitations: single primary_seed (13); symbolic uses fixed branch_a (expected
  value 0.5 by symmetry — the 0.48 reflects the realized correct-branch draws).

### EXP-FP-0004 — RESULT (2026-10-07)
- D: mean_return=-6.0000, stdev=0.0000 (all 200 steps, goal never reached).
  config_hash=c635edbfa785, receipt receipts/EXP-FP-0004-D.json
- fixed_predictor: mean_return=-5.9300, stdev=0.0691.
  config_hash=b030eeaf26cc, receipt receipts/EXP-FP-0004-S.json
- interpretation: NULL HOLDS (no meaningful difference; D trivially worse by
  0.07). Neither agent reaches the goal in 20 episodes. Predictor learning
  contributes nothing on pomaze under D's greedy policy — both agents wander.
  (Contrast: the symbolic baseline's frontier exploration reaches the goal in
  10/10 mazes, mean 43 steps — the task is solvable; D's policy is not.)
- limitations: single primary_seed (17); 20 episodes.

### Cross-experiment summary (2026-10-07)
D — the intended "null hypothesis with a pulse" — is weaker than intended:
it loses to the hand-coded symbolic baseline on every task where symbolic
applies (grid_world 0.82 vs -2.03; changing_rule WSLS 28 vs 20.6;
delayed_reward 0.48 vs 0.00; pomaze 10/10 solves vs 0/20), and barely
distinguishes itself from the weakest §39 baselines. Two precise, falsifiable
weaknesses were isolated: (1) the linear predictor cannot represent
cue×action contingencies (nonlinear interaction); (2) pure-greedy action
selection with no exploration locks into no-op fixed points. Both are
documented D limitations and concrete bars for architectures A/B/C:
beat D's numbers AND address (1) and (2), or the extra machinery is theater.
Negative results preserved in research/negative_results.md.

## EXP-FP-CALIB-01 — calibration battery for B's §30 predictions (§9 item 9)

**PREREGISTERED (2026-10-07, before run; full JSON: benchmarks/calibration_battery_preregistration.json)**
- hypothesis: Architecture B's §30 stated uncertainties are calibrated probabilistic claims: on all four domains the Brier score beats sequential climatology and the reliability curve sits near the diagonal.
- null: Brier_B >= Brier_climo on every domain (stated uncertainties carry no probabilistic information beyond the base rate).
- preregistered metric: per-domain Brier_B vs Brier_climo (sequential Laplace-smoothed climatology); ECE/MCE over 10 bins; signed error S. CALIBRATED iff Brier_B < Brier_climo AND ECE<=0.10 AND MCE<=0.25; MISCALIBRATED iff Brier_B >= Brier_climo or reliability fails; UNMEASURABLE on degenerate gates (n<200, std(p)<1e-9, base rate outside [0.02,0.98], non-finite).
- domains: D1 next-obs (o=1(mean|e0|<=0.2682), p=erf(EPS/(sigma*sqrt2)), sigma=ounc); D2 action-consequence (o=1(|rerr|<=0.2067), p=erf bridge, sigma=runc); D3 competence/failure (o=1(mean|e0|>0.4156), p=erfc bridge, sigma=ounc); D4 retrieval-usefulness (o=1(benefit>0), p=Phi(uhat/sigma_u)). Thresholds from 2340 PUBLISHED changing_rule ticks (pre-battery), frozen.
- baseline: sequential climatology (running base rate); secondary constant 0.5.
- conditions: changing_rule v1.0.0, arch_b, affect=none, 10 episodes x 5 fresh seeds {73101..73105}, PredictionLog JSONL per seed, canonical derive_seed streams.
- abstention: no threshold/bin/mapping tuning after the run; poor calibration everywhere is a negative-result finding, not a tuning trigger.
- config hash: recorded in receipt at run time.

**RESULT: pending — battery not yet run at registry time.**

**RESULT (2026-10-07, battery run complete — 5 fresh seeds, 1950 ticks/domain, receipt receipts/EXP-FP-CALIB-01.json)**
- D1 next_obs: MISCALIBRATED (fails to beat climatology; Brier 0.2532 vs climo 0.2508; ECE 0.048, MCE 0.909; corr(sigma,|err|)=0.02)
- D2 action_consequence: MISCALIBRATED (under-confident; Brier 0.2866 vs climo 0.2505; S=-0.19; corr=0.17)
- D3 competence_failure: MISCALIBRATED (over-confident; Brier 0.1025 vs climo 0.0734; S=+0.17; corr=0.02)
- D4 retrieval_usefulness: MISCALIBRATED (dispersion error; Brier 0.2610 vs climo 0.2472; observed benefit rate flat ~0.40-0.47 across predicted 0.08-0.64)
- Finding: B's stated uncertainties are essentially uncoupled from actual error magnitude; not decision-usable as probabilities. No post-run tuning per preregistered abstention rule. Full table: benchmarks/calibration_battery_results.json.

## EXP-FP-0006 — retrieval-usefulness gate: ûhat-gated vs unconditional vs random-gated on pomaze (§9 item 6, B gap #2)

**PREREGISTERED (2026-10-07, before run)**
- hypothesis: Gating the retrieval correction on ûhat (apply the correction only when uhat > 0, i.e. positive predicted error-reduction) beats unconditional retrieval-correction on pomaze mean episode return; AND beats random gating at the same application rate — ûhat carries ranking signal worth gating on.
- null: gated ≤ ungated, or gated ≤ random-gated — ûhat has no decision-usable ranking signal. Consistent with EXP-FP-CALIB-01 D4: observed benefit rate flat (~0.40–0.47) across the predicted range; corr(sigma,|err|)=0.05.
- preregistered metric (EXACT): mean episode return on pomaze over 15 closed-loop episodes. Per-seed: ΔGU = mean(G)−mean(U); ΔGR = mean(G)−mean(R). Win rule: seed-mean ΔGU > 0 AND seed-mean ΔGR > 0, with ≥3/4 seeds agreeing on the sign of BOTH comparisons.
- arms: U = ungated (current ArchB: correction applied whenever available; default gate_policy="ungated"). G = RetrievalGate(policy="uhat", threshold=0.0). R = RetrievalGate(policy="random", rate=r_s) where r_s = per-seed measured application rate of arm G (corrections applied / corrections available), gate_seed=derive_seed(s, 0, "gate"). R runs after U and G on the same seeds with r_s fixed per seed.
- baseline: arm U (unconditional retrieval-correction).
- conditions: pomaze v1.0.0, arch_b v1, affect='none', action_mode='active_inference', 15 episodes, identical primary seeds across the three arms (paired). 4 fresh seeds {73401, 73402, 73403, 73404} — never used by any prior experiment (Phase-3: 101–109; Phase-4: 301–333; repro: 72001–72705; calib: 73101–73105; consolidation: 7000s/900s).
- implementation: new module prototypes/architecture-b/retrieval_gate.py; the UsefulnessPredictor is NOT modified (gate reads uhat only). ArchB gains gate knobs (default "ungated" = bit-identical to current behavior); snapshot/restore extended for gate state.
- gate parameter: τ=0.0 frozen — apply iff predicted benefit positive. No tuning, per the abstention rule.
- closed-loop caveat (part of the tested object): when the gate blocks a correction, measured benefit = |e0_raw|−|e0_gated| = 0 and the predictor trains on that realized outcome. The experiment tests the full gated policy including its effect on predictor learning, not just the decision in isolation.
- secondary diagnostics (descriptive): gate rate r_s per seed; mean uhat for applied vs blocked decisions; corr(uhat, benefit) parsed from arm-U PredictionLog JSONL (unconditional data — the clean ranking check).
- verdict mapping: win rule met → H supported, gate INTEGRATED (maturity row updated). seed-mean ΔGR ≤ 0 → NEGATIVE: ûhat has no ranking signal; predictor stays EXECUTED, gap #2 closed by rejection, recorded in research/negative_results.md. ΔGU ≤ 0 but ΔGR > 0 → gating worse than unconditional but ûhat ranks (rate/threshold effect; investigate).

**RESULT: pending — run not yet executed at registry time.**

**RESULT (2026-10-07, run complete — 4 fresh seeds {73401..73404}, 15 pomaze episodes/arm/seed, receipt receipts/EXP-FP-0006.json, hash-chained)**
- The ûhat sign gate COLLAPSED to never-apply on all 4 seeds: application rate 0.000 (n_available 2621–2999, applied 0). Cold-start degeneracy — uhat inits at exactly 0.0, strict `>` blocks from tick 0, predictor trains on realized benefit=0, uhat stays exactly 0.0. The policy is not self-bootstrapping.
- Per-seed: U −3.757/−4.001/−3.599/−3.244; G −3.526/−3.897/−3.924/−3.093; R identical to G (rate 0 → never-apply). ΔGU +0.231/+0.104/−0.325/+0.151 (seed-mean +0.040); ΔGR 0.000 on all 4 (G≡R, vacuous comparison).
- Preregistered rule fires NEGATIVE (ΔGR ≤ 0) — but the rejection is of the closed-loop sign-gate policy (it never made a discriminating decision), NOT of ûhat's ranking ability.
- Clean ranking check (arm-U PredictionLogs, unconditional corrections, in-sample): corr(uhat, benefit) = +0.403…+0.483 across 4 seeds (n≈2600–3000), P(benefit>0)≈0.67, mean uhat tracks mean benefit. The predictor learns a real ranking signal; this gate cannot exploit it.
- Secondary: skipping vs applying the correction shows no measurable return difference on pomaze (ΔGU −0.33…+0.23 per seed).
- Predictor stays EXECUTED. The ranking-exploitation question needs a non-degenerate instrument (shadow-trained predictor, warm start, non-strict threshold) — future experiment, not a retrofit.

---

## EXP-FP-0005 — consolidation race: S-01 vs memory_port, prioritized vs uniform vs no offline replay (§9 item 8)

**PREREGISTERED (2026-10-07, before run; full spec sealed in prototypes/architecture-b/consolidation_race.py module docstring)**
- hypothesis: Offline prioritized consolidation replay causally improves later prediction performance on a held-out probe vs no replay, and beats uniform replay at equal budget.
- null: Offline replay changes nothing (no arm beats N); or prioritized replay does not beat uniform (priority is decoration).
- preregistered metric: IG_probe = probe_before − probe_after, probe error = mean|eval_transition| on a FIXED 400-transition probe set (random-policy pomaze rollouts, probe seed 901, policy-independent, identical across arms/seeds). Fresh ArchB per arm, identical init (seed 7000+s), one online pass over the corpus before the offline phase. Higher IG = larger offline improvement.
- arms: P1 prioritized via S-01 (promotion list, members in (-seeded relevance, id) order); P2 prioritized via memory_port (consolidate + get_replay_batch top-200); U uniform 200 without replacement (own mulberry32 PRNG, seed 4242+s — no random.* module, tripwire-clean); N no replay.
- corpus: ONE fixed corpus per seed for all arms — 24 closed-loop pomaze episodes from fixed reference ArchB (active_inference, affect='none', seed 900+s).
- relevance (preregistered; deployment-specific per H5 risk): rel_i = max(0.2, min(1.0, pe_i/max_pe)), pe_i = post-online eval_transition error. Identical for P1/P2.
- replay budget K=200 per replay arm.
- seeds: {61701, 61702, 61703, 61704} — fresh, no overlap with battery/Phase-4/repro seeds.
- decision rules: G1: arm beats N iff seed-mean IG higher AND >= 3/4 per-seed wins (separate for P1, P2). G2: same rule P1-vs-U, P2-vs-U. G3: implementation winner = argmax(seed-mean IG) over {P1,P2}; no-winner if |Δ|<1e-6. If G1 fails for BOTH P1 and P2: Phase-4 battery claim gets a BOUND (mechanism CAUSAL, no measured offline lift on this corpus) — negative result.
- gates (VOID, not reinterpreted): G0a corpus >= 300; G0b probe_before > 0; G0c replay counts exactly 200 in P1/P2/U; G0d determinism spot-check (seed 61701 P1 recompute, IG to 1e-12); G0e fabrication-tripwire CLEAN (done pre-run); G0f hash-chained receipt + verify_chain.
- conditions: pomaze v1.0.0, arch_b v1, affect='none', contract v1.0.0.
- config hash: recorded in receipt at run time.

**AMENDMENT (2026-10-07, pre-interpretation — first run declared VOID per G0c)**
- What happened: the first run completed all gates except G0c — memory_port's merge at the preregistered threshold 0.9 collapsed the ~4549-transition pomaze corpus to ~17–21 consolidated entries, so the raw top-200 replay batch yielded only 17/21/19/21 items (one value per seed).
- Ruling: run VOID, per the preregistered gate. No numbers from the voided run were interpreted or retained for any claim; its results file was discarded.
- Amended P2 replay protocol (implementation code AND parameters untouched): multiplicity replay through the consolidated entries — each entry replayed merged_count times (one training update per source transition, routed through the entry's consolidated mean vector with the representative member's (a, o2, r) — the merge carries non-vector fields from the highest-relevance member), in batch relevance order, total capped at 200 updates. This races memory_port's own "consolidate in place, replay the compacted entries" philosophy while restoring budget parity with P1/U.
- Rerun on the same seeds {61701–61704} (deterministic; voided numbers discarded).

**RESULT (2026-10-07, amended race run complete — 4 fresh seeds {61701..61704}, receipt receipts/EXP-FP-0005-S.json + EXP-FP-0005-D.json, all gates passed)**
- Race table (IG_probe = probe_before − probe_after; higher = larger offline improvement):

  | seed | probe_before | P1 (S-01) | P2 (memory_port) | U (uniform) | N (none) |
  |---|---|---|---|---|---|
  | 61701 | 0.877786 | −0.111382 | −0.159458 | +0.211213 | 0.0 |
  | 61702 | 0.731743 | −0.245686 | −0.451768 | +0.066894 | 0.0 |
  | 61703 | 1.070455 | +0.101716 | −0.029023 | +0.373831 | 0.0 |
  | 61704 | 1.272748 | +0.451196 | +0.383111 | +0.641678 | 0.0 |
  | seed-mean | — | **+0.049** | **−0.064** | **+0.323** | 0.0 |

- G1 (replay vs no replay): P1-vs-N 2/4 per-seed wins (seed-mean +0.049 > 0) — FAILS the ≥3/4 rule. P2-vs-N 1/4 — FAILS. **Neither prioritized arm beats no-replay.**
- G2 (prioritized vs uniform): U beats P1 4/4, U beats P2 4/4 (U seed-mean +0.323). Prioritization by prediction-error magnitude consistently HURTS offline replay on this corpus.
- G3 (implementation race): **P1 (S-01) beats P2 (memory_port)** — seed-mean IG +0.049 vs −0.064 (Δ=+0.113); P1 > P2 on 4/4 seeds. S-01's "promote ids, replay raw" outperforms memory_port's "replay compressed summaries" here.
- Interpretation: the Phase-4 battery "consolidation CAUSAL" claim gets a BOUND — the consolidation mechanism behaves per docs (mechanism CAUSAL, per brothel EXP-MEMORY-001), but offline replay shows NO measured probe improvement vs no-replay on the pomaze corpus, and prioritization is actively worse than uniform replay. Recorded as a negative result (research/negative_results.md). Mechanistic note: both implementations merge ~4600 transitions into ~17–21 groups at threshold 0.9 — pomaze obs vectors are highly mutually similar.
- Limitations: pomaze only; priority function is prediction-error magnitude (deployment-specific); budget K=200 updates (~4% of corpus); probe measures model prediction error, not closed-loop return.

## EXP-AB-K3B — hierarchy earns its keep on longer structured tasks (§H5)

**PREREGISTERED (2026-10-07, before run; spec sealed in the detail receipt's `preregistered` block)**
- hypothesis: With longer training on structured tasks, the intact-vs-L1-lesioned prediction gap GROWS beyond K3's +0.0032: the top-down context experts earn their keep where predictable structure spans longer horizons.
- null: Gaps stay tiny/absent — the hierarchy effect is bounded at K3's weak level.
- preregistered metric: Arm A (changing_rule, 1500 train transitions, held-out 200 from a stable phase): mean|e0| and mean|rerr| intact vs lesioned. Arm B (delayed_reward, 600 train / 120 held-out): same metrics; preregistered expectation: terminal |rerr| gap ≈ 0 because the correct branch is hidden (50/50 unpredictable from obs) — arm B bounds the test, arm A carries it.
- ablation: L1 top-down path (lesion_l1) — same lesion as K3.
- baseline: arch_b lesion_l1=True (L0-only), identical transitions.
- conditions: affect 'none', arch_b v1; open-loop transition training via learn_transition (no retrieval correction/affect — isolates the weight/precision machinery, as in K3).
- seeds: {331, 332, 333} (fresh; 3 seeds).
- decision rule: Arm-A |e0| or |rerr| lesion gap clearly larger than K3's +0.0032 (seed-mean) -> hierarchy earns its keep. Gaps at K3 scale or smaller -> effect stays weak; note as a bound.
- config hash: a0ef664d58f69009f3527abe88e56e474b31dae85c705ac1f8f577d076859e72 (recorded in receipt at run time).

**RESULT (2026-10-07, receipt prototypes/architecture-b/receipts/EXP-AB-K3B.json; written 2026-10-07T07:00:35Z)**
- Arm A (changing_rule), lesion gaps = lesioned − intact (seed-means over 3 seeds):
  |e0| +0.0054 (per-seed: −0.0109, +0.0104, +0.0168); |rerr| +0.0602 (per-seed: +0.0791, +0.0483, +0.0532). n_train=1500, n_heldout=200 per seed.
- Arm B (delayed_reward), seed-means: |e0| +0.0437 (per-seed: +0.0352, +0.0451, +0.0508); |rerr| −0.0002 (≈0, as expected); terminal |rerr| +0.0073 (per-seed: +0.0220, 0.0, 0.0 — seeds 332/333 had n_terminal=0). n_train=600, n_heldout=120.
- Decision rule applied: Arm-A |rerr| gap +0.0602 is ~19x K3's +0.0032 — **hierarchy earns its keep** on structured tasks with longer horizons.
- Interpretation (from receipt): "HIERARCHY EARNS ITS KEEP: arm-A lesion gaps grew vs K3's +0.0032 — |e0| gap +0.0054, |rerr| gap +0.0602 (seed-means). Arm B: |e0| +0.0437, terminal |rerr| +0.0073 (expected ≈0 — hidden branch bounds it)."
- Limitations (from receipt): "Open-loop training isolates the weight/precision machinery (as K3); closed-loop control effects are factored out by design. Arm-B terminal ticks flagged by reward >= 0.5. 3 seeds."
- Receipts: prototypes/architecture-b/receipts/EXP-AB-K3B.json (detail only — no lane-summary receipt was written to receipts/). The detail receipt is pre-chain (no receipt_hash) as of this writing.

## EXP-AB-K3C — does the hierarchy effect generalize to harder tasks (§H5)

**PREREGISTERED (2026-10-07, before run; from the lane-summary receipt)**
- hypothesis: B's hierarchy (L1 context-expert top-down path) earns its keep beyond the two K3B envs: (task 1) on a longer multi-stage delayed task the intact-vs-L1-lesioned |e0| gap stays positive; (task 2) on a compositional XOR contingency the intact-vs-lesioned |rerr| gap stays positive.
- null: Lesion gaps vanish (<= 0) on the harder tasks — the hierarchy effect is a two-env phenomenon.
- preregistered metric: Per task: lesion gap D = lesioned − intact on held-out transitions. Task 1 (delayed_multistep) carrying: D_e0. Task 2 (compositional_rule) carrying: D_rerr.
- baseline: arch_b lesion_l1=True (L0-only), identical transitions.
- conditions: affect 'none', arch_b v1, contract 1.0.0. Task 1: delayed_multistep v1.0.0, 30 episodes (train eps 0–19, held-out eps 20–29), scripted reference policy. Task 2: compositional_rule v1.0.0, 45 episodes (train eps 0–39, 1600 transitions; held-out eps 40–44, 200 transitions), ArchB closed-loop reference.
- primary_seed: '74101-74104' (4 fresh seeds).
- config hash: a922e3e25ce3cf55850b006cc8f4954d9f38abe8ecae3289d1d30e789adcdded (recorded in receipt at run time).

**RESULT (2026-10-07, lane-summary receipt receipts/EXP-AB-K3C.json, written 2026-10-07T08:49:50Z; detail receipt prototypes/architecture-b/receipts/EXP-AB-K3C.json, written 2026-10-07T08:49:49Z)**
- Task 1 (delayed_multistep), D_e0 seed-mean −0.0470 (per-seed: −0.0587, −0.0503, −0.0426, −0.0363); D_rerr seed-mean −0.2134; terminal D_rerr +0.0999. passes_gate: false.
- Task 2 (compositional_rule), D_rerr seed-mean −0.2684 (per-seed: −0.2172, −0.3406, −0.2473, −0.2686); D_e0 seed-mean −0.0097. passes_gate: false.
- Verdict: **NO EFFECT** — lesion gaps vanish on both new tasks.
- Interpretation (from receipt): "NO EFFECT on harder tasks: lesion gaps vanish on both new tasks (task1 D_e0 seed-mean -0.0470, task2 D_rerr seed-mean -0.2684). The hierarchy effect is a two-env phenomenon — K3B's gaps do not generalize."
- Limitations (from receipt): "Open-loop training isolates the weight/precision machinery (as K3/K3B); closed-loop control effects are factored out by design. Task-1 reference is a scripted policy (ArchB/random references stall pe..." (verbatim, truncated in receipt).
- Negative result: recorded as a bound on the K3B claim — the K3B hierarchy effect does not generalize beyond the two K3B envs.
- Receipts: receipts/EXP-AB-K3C.json (lane summary, chained) + prototypes/architecture-b/receipts/EXP-AB-K3C.json (detail, pre-chain as of this writing). Note: the lane-summary receipt was written while the old harness write/verify convention was in force (prev_receipt_hash=None); this historical byte state is preserved untouched per lab law.

---

## EXP-FP-0007 — non-degenerate ûhat-gating instrument: shadow-trained predictor (§9 item 6, B gap #2 follow-up)

**PREREGISTERED (2026-10-07, before run — code not yet written)**
- hypothesis: Gating the retrieval correction on ûhat from a
  shadow-trained predictor (training data uncontaminated by gating) beats
  unconditional retrieval-correction on pomaze mean episode return, AND
  beats random gating at the same application rate — the learned ranking
  signal (corr(uhat, benefit) = +0.40…+0.48 on clean unconditional data,
  EXP-FP-0006) carries decision-usable value once the cold-start
  degeneracy is removed.
- null: gated ≤ ungated, or gated ≤ random-gated — the ranking signal has
  no decision value even with a non-degenerate instrument.
- preregistered metric (EXACT): mean episode return on pomaze over 15
  closed-loop episodes. Per-seed: ΔGU = mean(G)−mean(U);
  ΔGR = mean(G)−mean(R). Win rule: seed-mean ΔGU > 0 AND seed-mean
  ΔGR > 0, with ≥3/4 seeds agreeing on the sign of BOTH comparisons.
  (Identical to EXP-FP-0006.)
- arms: U = ungated (correction applied whenever available;
  gate_policy="ungated"). G = shadow-ûhat gate (gate_policy="uhat",
  threshold=0.0 FROZEN, gate_uhat_source="shadow"): the gate reads ûhat
  from a second UsefulnessPredictor updated on every available-correction
  tick with the counterfactual unconditional benefit
  |obs−xhat_raw|−|obs−(xhat_raw+correction)|; the live predictor trains on
  realized benefit exactly as in EXP-FP-0006 and is not read by the gate.
  The UsefulnessPredictor class itself is NOT modified (gate reads only).
  R = RetrievalGate(policy="random", rate=r_s) where r_s = per-seed
  measured application rate of arm G, gate_seed=derive_seed(s, 0, "gate").
  R runs after U and G on the same seeds.
- baseline: arm U (unconditional retrieval-correction).
- conditions: pomaze v1.0.0, arch_b v1, affect='none',
  action_mode='active_inference', 15 episodes, identical primary seeds
  across the three arms (paired). 4 fresh seeds {73501, 73502, 73503,
  73504} — no overlap with any prior experiment's seeds.
- instrument selection (recorded before implementation): option (a)
  shadow-trained predictor CHOSEN over (b) warm start — a warm-started
  predictor retrains on gated/contaminated outcomes and drifts back toward
  the degenerate fixed point, so the instrument would still test a
  contaminated policy; and over (c) non-strict threshold + warm start —
  uhat>=0 at cold start applies nearly everything, muddying whether any
  win comes from ranking or from a near-ungated rate. The shadow isolates
  exactly one variable: the ranking signal's decision value with
  uncontaminated training data.
- frozen gates (VOID, not reinterpreted): G0 instrument-validity — G
  application rate strictly in (0, 1) on ≥3/4 seeds (the gate must make
  discriminating decisions); else VOID as instrument failure. This is the
  EXP-FP-0006 lesson: distinguish "gate collapsed" from "ranking has no
  value". G1 fabrication-tripwire CLEAN on new/modified code, pre-run.
  G2 determinism spot-check: seed 73501 G arm recomputed → mean_return
  identical to 1e-12, else VOID. G3 hash-chained receipt + verify_chain.
- closed-loop caveat (part of the tested object): the gated trajectory
  diverges from the ungated trajectory once gating blocks; the shadow
  predictor's training data is uncontaminated by gating (counterfactual
  unconditional benefit) but lives on the gated trajectory. The experiment
  tests the full shadow-gated policy.
- honest prior: EXP-FP-0006's secondary found skipping vs applying the
  correction return-neutral on pomaze (ΔGU −0.33…+0.23 per seed) — a NULL
  on return remains plausible even with a working instrument. The value is
  in testing the instrument and bounding the ranking signal's decision
  value.
- secondary diagnostics (descriptive): G application rate per seed;
  corr(uhat_shadow, benefit_shadow) on the G arm; corr(uhat_live,
  benefit_realized) from the U arm PredictionLog (replication of the
  EXP-FP-0006 ranking check on fresh seeds).

**RESULT: pending — preregistered before implementation.**

**AMENDMENT (2026-10-07, pre-interpretation — first run declared VOID per G2)**
- What happened: the first run completed all four seeds (G application
  rates 0.79–0.80, non-degenerate; shadow corr +0.40…+0.53) but the G2
  determinism spot-check FIRED — the recomputed seed-73501 G arm's
  mean_return differed from the stored value by more than 1e-12.
- Root cause (verified, not assumed): the G2 gate compared the recompute's
  UNROUNDED mean_return against the per-seed dict's ROUNDED (4-decimal)
  G_mean_return — a gate implementation bug, not experimental
  nondeterminism. Direct check: two fresh G-arm runs on seed 73501 give
  bit-identical means (-2.4593333333333214 == -2.4593333333333214).
- Ruling: run VOID per the frozen gate. No numbers from the voided run
  were interpreted or retained for any claim.
- Fix (code only, preregistration untouched): the per-seed dict now
  carries G_mean_return_raw (unrounded) and G2 compares unrounded at
  1e-12, honoring the preregistered precision.
- Rerun on the same preregistered seeds {73501–73504} (deterministic;
  voided numbers discarded).

**RESULT (2026-10-07, rerun complete — 4 fresh seeds {73501..73504}, 15
pomaze episodes/arm/seed, receipt receipts/EXP-FP-0007.json, hash-chained;
own hash verifies, prev linkage correct; historical chain issues
(pre-chain receipts, K3C old-convention prev) preserved untouched per lab
law)**
- The shadow instrument is NON-DEGENERATE: G application rate
  0.791–0.800 on all 4 seeds (n_available 2675–2941, discriminating
  decisions throughout). G0 passes 4/4. G2 determinism spot-check PASSES
  (unrounded compare at 1e-12). G1 tripwire clean, G3 receipt chained.
- Per-seed: U −3.189/−3.875/−3.537/−3.144; G −2.459/−3.567/−3.550/−2.959;
  R −3.029/−3.581/−3.616/−3.321. ΔGU +0.729/+0.308/−0.013/+0.185
  (seed-mean +0.3023, 3/4 agree); ΔGR +0.570/+0.015/+0.066/+0.362
  (seed-mean +0.2532, 4/4 agree).
- Preregistered win rule FIRES: **H SUPPORTED** — the shadow-ûhat gate
  beats unconditional retrieval-correction AND beats random gating at the
  matched rate. The learned ranking signal carries decision-usable value
  once the cold-start degeneracy is removed.
- Shadow diagnostics: corr(uhat_shadow, benefit_shadow) on the G arm =
  +0.399…+0.533 (n≈2700–2940) — the shadow predictor learns the same
  ranking the live one does, on uncontaminated data. U-arm replication of
  the EXP-FP-0006 ranking check: corr(uhat, benefit) = +0.405…+0.464 —
  the signal is stable across 8 seeds total.
- Honest bounds: all returns remain negative (pomaze unsolved by all
  arms, as in EXP-FP-0006) — the gate improves return by ~+0.30 on a
  −3.5 baseline, a real but modest effect; seed 73503 ΔGU is −0.013
  (essentially zero, not a loss). Single env (pomaze), 4 seeds.
- Maturity: gate policy PROMOTED to INTEGRATED (wired, non-degenerate,
  measured behavioral value vs no-gate and chance-gate). CAUSAL label
  withheld pending replication breadth — the evidence is causal-shaped
  but single-env.
- Predictor stays EXECUTED (unchanged class); the gate is the integrated
  consumer of its ranking signal. B gap #2 now has a working instrument
  and a positive result, replacing the EXP-FP-0006 rejection-of-policy.

---

## EXP-FP-0008 — precision-explosion characterization: estimated vs uniform reward head on rare terminal spikes (C2 pathology family follow-up)

**PREREGISTERED (2026-10-07, before run — code not yet written; full JSON:
experiments/preregistration_EXP-FP-0008.json)**
- background: EXP-AB-K3C task-1 pilot (throwaway seed 99901,
  delayed_multistep): the lesioned model's estimated-precision reward head
  exploded on rare terminal +1.0 spikes (5/600 train transitions) —
  b_r=0.31, max|w_r|=0.73, rhat=1.27 predicted on a 0.02 tick; held-out
  mean|rerr|=0.8256 vs intact 0.2809; lesioned+uniform control stayed sane
  (b_r=0.018, mean|rerr|=0.10). Mechanism hypothesis: rare spikes in a ~0
  reward sea lower rerr variance → estimated piR stays high → massive
  per-spike w_r updates (d = eta_r·piR·rerr·f) with no absorber. Second
  independent sighting of the C2 precision pathology family (C2 kill was
  FRAGILE 3/5).
- hypothesis: The estimated-precision reward head systematically
  misbehaves on rare high-reward events: across tasks with rare terminal
  spikes, intact ArchB with estimated precision shows the explosion
  signature on terminal-spike trials while the uniform-precision control
  stays sane; on a dense-reward task the signature is absent.
- null: no systematic estimated-vs-uniform difference on terminal-spike
  trials across tasks.
- preregistered metric (EXACT): per task per seed, on held-out
  terminal-spike trials (reward ≥ 0.5): explosion signature iff
  mean|rerr|_estimated_terminal ≥ 3 × mean|rerr|_uniform_terminal AND
  mean|rerr|_estimated_terminal > 0.3. Task signature iff ≥3/4 seeds show it.
- verdict mapping (CHARACTERIZATION, not a kill): SYSTEMATIC iff ≥2 of 3
  tasks show the task signature; TASK-LOCAL iff exactly 1; ABSENT iff 0.
- tasks: pomaze v1.0.0 (rare goal +1.0), delayed_reward v1.0.0 (delayed
  +1.0 at corridor end), changing_rule v1.0.0 (dense {0,1}, negative
  control — signature expected ABSENT; rewards are not rare so rerr
  variance stays high and piR stays low).
- procedure (K3B-instrument-faithful): per task per seed: (1) collect a
  transition stream with the symbolic reference baseline (pomaze: frontier
  exploration; delayed_reward: branch_a + forward; changing_rule: WSLS) —
  train episodes then held-out episodes with disjoint episode seeds;
  (2) two INTACT ArchB (lesion_l1=False) with IDENTICAL init seeds train
  open-loop via learn_transition on the identical train stream — arm E
  estimated precision (default), arm F uniform_precision=True;
  (3) held-out eval via predict_reward → |rhat−r|; terminal-spike trials
  flagged reward ≥ 0.5 (same flag as K3B/K3C).
- episode counts: pomaze train 20 / held 10; delayed_reward train 30 /
  held 15; changing_rule train 30 / held 10.
- seeds: {74301, 74302, 74303, 74304} — fresh, no overlap with any prior
  experiment's seeds. Train/eval init seeds: derive_seed(s, 0, "agent").
- secondary diagnostics (descriptive, mechanism check): post-training b_r
  and max|w_r| per arm; mean piR on terminal training trials (arm E);
  training-stream rerr variance (arm E); realized spike rate per task.
- frozen gates (VOID, not reinterpreted): G0a held-out terminal trials ==
  0 for a task → VOID that task. G0b fabrication-tripwire CLEAN pre-run.
  G0c determinism spot-check: seed 74301 arm E recomputed → b_r identical
  to 1e-12, else VOID. G0d hash-chained receipt + verify_chain.
- limitations (preregistered): open-loop training isolates the
  weight/precision machinery (K3/K3B/K3C precedent); closed-loop control
  effects factored out by design. Intact models only — the pilot sighting
  was lesioned; if the signature needs the lesion's missing R_ctx absorber
  this design will show TASK-LOCAL/ABSENT and that boundary is itself the
  finding.

**RESULT: pending — preregistered before implementation.**

**AMENDMENT (2026-10-07, pre-interpretation — first run declared VOID per G0c)**
- What happened: the first run completed all 3 tasks × 4 seeds but the
  G0c determinism spot-check FIRED on the seed-74301 pomaze arm-E
  recompute.
- Root cause (verified, not assumed): the G0c gate compared the
  recompute's UNROUNDED b_r against the per-seed dict's 6-decimal-ROUNDED
  b_r — a gate implementation bug (the same rounded-vs-unrounded class
  of bug as EXP-FP-0007's voided first run), not experimental
  nondeterminism. Direct check: two fresh arm-E runs give bit-identical
  b_r (0.3653066432560002 == 0.3653066432560002).
- Ruling: run VOID per the frozen gate. No numbers from the voided run
  were interpreted or retained for any claim.
- Fix (code only, preregistration untouched): the per-seed dict now
  carries E_b_r_raw (unrounded) and G0c compares unrounded at 1e-12.
- Rerun on the same preregistered seeds {74301–74304} (deterministic;
  voided numbers discarded).

**RESULT (2026-10-07, rerun complete — 4 fresh seeds {74301..74304}, 3
tasks × 2 arms, receipt receipts/EXP-FP-0008.json, hash-chained; own hash
verifies, chained to EXP-FP-0007; G0a/G0c/G0d pass, no voided tasks)**
- Verdict per the preregistered mapping: **ABSENT** — 0/3 tasks reach the
  ≥3/4-seed task-signature bar. pomaze 1/4 (seed 74303: E_term=3.4191 vs
  F_term=0.7807, ratio 4.38 — single-seed outlier, not a pattern; other
  seeds ratios 0.30–1.24); delayed_reward 0/4 (ratios 0.82–1.36);
  changing_rule 0/4 (ratios 1.00–1.16 — negative control behaves as
  expected).
- Mechanism secondaries: the weight-inflation half of the pathology DOES
  operate on intact models — b_r / max|w_r| inflate vs uniform on the
  spike tasks (pomaze max|w_r| E 0.75–1.61 vs F 0.33–0.35; delayed_reward
  E 0.73–1.17 vs F 0.17–0.25). But R_ctx context tables absorb it:
  estimated piR on terminal training trials sits at 3.3–5.4 (well below
  pi_max=20, not pinned), and held-out terminal |rerr| does not
  systematically explode. Realized spike rates: pomaze ~0.023–0.028,
  delayed_reward ~0.033–0.043, changing_rule ~0.78–0.80 (dense, as
  designed).
- Interpretation: the C2 pathology family is BOUNDED — the pilot's full
  explosion needed the lesion's missing R_ctx absorber (or the
  delayed_multistep structure). Estimated precision misbehaves in
  specific configurations, not universally; consistent with C2's FRAGILE
  3/5 kill. Recorded as NR-B-010 (characterization bound, not a kill).
- Limitations: intact models only (preregistered boundary); open-loop
  training isolates weight/precision machinery; symbolic reference
  streams; delayed_reward held-out terminal trials thin (4–8/seed) but
  nonzero — G0a passes.

---

## EXP-AB-5SEED — 5-seed multi-seed replication of Phase-4 (M1, C2B, C4B, K3B) (§9 item #3)

**PREREGISTERED (2026-10-07, 09:06 UTC, before run; full JSON: experiments/preregistration_EXP-AB-5SEED.json)**
- instrument: VERBATIM reuse of prototypes/architecture-b/experiments_phase4.py (exp_m1, exp_c2b, exp_c4b, exp_k3b) — no redefinition, no parameter changes. Only the seed-list parameter changes (3 -> 5 fresh seeds) and receipt IDs (repro5_EXP-AB-*); additive driver prototypes/architecture-b/repro_phase4_5seed.py.
- replication rule (frozen): verdict REPRODUCES / KILL REPLICATES iff the original verdict's per-seed decision holds on >= 4/5 fresh seeds. < 4/5 -> FRAGILE (3/5) or OVERTURNED (<= 2/5).
- repro5_EXP-AB-M1: replicates EXP-AB-M1 (SELECTIVE DEFICIT: pomaze +2.200 vs delayed_reward +0.046). Per-seed selective = delta_pom > 0 AND delta_pom > delta_drw + 1e-9. Fresh seeds {75101..75105}.
- repro5_EXP-AB-C2B: replicates EXP-AB-C2B (KILL: shift_reset 23.97 < estimated 24.10 < uniform 28.47; stationary 24.40 < 34.00). Per-seed revival = (shift: shift_reset >= estimated − 1e-9 AND > uniform + 1e-9) AND (stationary: shift_reset >= uniform − 1e-9). KILL REPLICATES iff revival fails on >= 4/5. Fresh seeds {75201..75205}.
- repro5_EXP-AB-C4B: replicates EXP-AB-C4B (KILL PERMANENT: AI 0.5443 < greedy 0.7124 ≈ random 0.5497). Per-seed kill-holds = AI IG_probe <= max(greedy, random) + 1e-9. KILL REPLICATES iff >= 4/5. Fresh seeds {75301..75305}.
- repro5_EXP-AB-K3B: replicates EXP-AB-K3B (SURVIVES, STRENGTHENED: arm-A |rerr| gap +0.0602 3/3). Per-seed grown = delta_e0 > 0.0032 + 1e-9 OR delta_rerr > 0.0032 + 1e-9 on arm A. REPRODUCES iff >= 4/5. Fresh seeds {75401..75405}.
- seed freshness: all 20 primaries checked against the registry, all receipts, and all driver files — no overlap with phase3 (101-109), phase4 (301-333), repro (72001-72705), calib (73101-73105), consolidation (61701-61704), retrieval-gate (73401-73404), K3C (74101-74104), pathology (74301-74304).
- frozen gates: G0a determinism spot-check (M1 seed 75101 re-run through identical helper calls, match to 1e-12, else VOID). G0b seed-level VOID on crash/non-finite (frozen list, no reseeding). G0c tripwire CLEAN pre-run (done); gate suite green before close. No post-run tuning.

**RESULT (2026-10-07, all four complete — identical instruments, receipts hash-chained)**

- repro5_EXP-AB-M1 (seeds {75101..75105}, G0a determinism PASS): selective deficit holds 5/5 — pomaze deltas +2.7987/+2.2547/+1.8973/+2.1273/+1.7120 (seed-mean +2.158), delayed_reward +0.0860/+0.0420/+0.0460/+0.0440/+0.0300 (seed-mean +0.050). **REPRODUCES 5/5.** EpisodicStore earns REPRODUCED — first arch-B component at that maturity. Receipts: receipts/repro5_EXP-AB-M1.json + prototypes/architecture-b/receipts/repro5_EXP-AB-M1.json.
- repro5_EXP-AB-C2B (seeds {75201..75205}): shift arm seed-means shift_reset 25.60 vs estimated 24.36 vs uniform 28.68; stationary shift_reset 26.60 vs uniform 33.28. Revival holds on 0/5 seeds. **KILL REPLICATES 5/5.** Receipts: receipts/repro5_EXP-AB-C2B.json + prototypes/architecture-b/receipts/repro5_EXP-AB-C2B.json.
- repro5_EXP-AB-C4B (seeds {75301..75305}): AI IG_probe 0.4476 vs greedy 0.6487 vs random 0.5530 (seed-means); kill condition holds 5/5. **KILL REPLICATES 5/5.** Driver labeling bug caught at close: the verdict map inverted C4B's holds-count and first wrote KILL OVERTURNED; corrected to KILL REPLICATES in both receipts with an amendment note, receipts re-hashed — per-seed data untouched. Receipts: receipts/repro5_EXP-AB-C4B.json + prototypes/architecture-b/receipts/repro5_EXP-AB-C4B.json.
- repro5_EXP-AB-K3B (seeds {75401..75405}): per-seed "grown" holds 4/5 (fails on 75402). **Preregistered gate fires REPRODUCES at exactly 4/5 — BUT the carrying channel reversed**: arm-A |rerr| gap −0.0319 seed-mean, negative on 3/5 seeds (75402 −0.0305, 75403 −0.0855, 75404 −0.2753); the signal rides |e0| (+0.0079, positive 4/5) at ~2.5× K3 scale. The +0.0602 was seed-fragile. Verdict stands per the frozen gate; substance weakened (NR-B-009, research/negative_results.md). Receipts: receipts/repro5_EXP-AB-K3B.json + prototypes/architecture-b/receipts/repro5_EXP-AB-K3B.json.
- interpretation: pure prediction learning (K2) and episodic memory (M1) replicate cleanly; the adaptive machinery (C2/C2B, C4/C4B) stays dead on fresh seeds; the hierarchy's longer-horizon signal replicates in the letter of the gate but loses its headline channel. The cross-cutting pattern holds: prediction machinery learns robustly; context/precision machinery does not robustly convert learning into better decisions.
- limitations: pomaze/changing_rule/delayed_reward only; open-loop for K3B; C2B's ONE variant only; G0a spot-check covered M1 only (identical seeded-RNG pipeline shared by all four instruments).

---

## EXP-SW-01 — self/world distinction probe (§9 item #12)

**PREREGISTERED (2026-10-07, 09:07 UTC, before implementation; full JSON: experiments/preregistration_SELF_WORLD.json; 2 pre-run amendments, no experimental seeds touched)**
- new env (additive): self_world v1.0.0 — hand channel SELF-caused (deterministic f(action)), ball channel WORLD-caused (exogenous fair coin); STEP=0.2, MAX_STEPS=60, reward 0.0; matched change statistics (same step size, symmetric reflection, 50/50 directions); cause labels in info for SCORING ONLY (agents never see info).
- B arm: ArchB affect='none', open-loop learn_transition training (10 eps x 60), held-out predict_next eval (5 eps x 60); intact vs frozen (paired streams + paired init seeds 75001..75004). Carrying metric (AMENDED pre-run): D_adj = D_intact − D_frozen per seed, D = mean|e_ball| − mean|e_hand|, identical held-out stream for both arms (the amendment: throwaway-seed pilot showed the frozen arm also gaps positive from position-marginal sampling noise — D_frozen is stream-specific bias, so it is subtracted paired rather than bounded absolutely).
- A arm: WorkspaceTick closed-loop via scalar-reward adapter, change-detector specialists, 4 eps x 60 ticks. Carrying: Delta_bid, Delta_ign; DISTINCTION iff |seed-mean| > 0.05 on either with consistent sign >= 3/4 seeds. Expectation: NO DISTINCTION (honest negative — no action-conditioned path in A). Delta_gain diagnostic only (tie-break + acted-channel-only utility updates = credit-assignment artifact).
- frozen gates: G0a held-out transitions == 0 -> VOID arm; G0b determinism check -> VOID run. Tripwire CLEAN pre-run on new code.
- seeds {75101..75104}. Note: seed integers coincide with the concurrent repro5_EXP-AB-M1 lane's {75101..75105} (preregistered 09:06 UTC, one minute earlier, unknown to this worker at design time) — NO contamination: different envs, different models, domain-separated RNG streams; reproducibility is per (experiment, seed).

**RESULT (2026-10-07, ~09:35 UTC)**
- B: **DISTINCTION DETECTED (PASS)** — seed-mean D_adj = +0.091 (> +0.05), D_adj > 0 on 4/4 seeds (+0.019/+0.174/+0.029/+0.141). D_intact stable 4/4 (+0.109…+0.128): intact e_hand ≈ 0.05–0.06 vs e_ball ≈ 0.16–0.19. Per-seed D_adj variance is driven by D_frozen noise (−0.053…+0.090) — the paired correction working as designed. Interpretation: B's action-conditioned generative model predicts self-caused changes better than world-caused ones; the prediction-error gap is learned (frozen shows only stream noise), i.e. B carries the self/world distinction in its prediction error. Receipt: receipts/EXP-SW-01-B.json (hash-chained, self-verified).
- A: **NO DISTINCTION (null holds — expected negative)** — seed-mean Delta_bid = +0.0009, Delta_ign = 0.000 (nothing ever ignited: habituated bids sit below the ignition threshold on both channels). Gains floored symmetric (0.01/0.01). Interpretation: no internal variable of A distinguishes matched self/world changes — A's specialists/arbitrator/ignition have no action-conditioned path (no efference copy) and the z-score bid habituates per channel to 0.5 regardless of cause. Instrument validated by unit test (chain discriminates unequal stimuli). Receipt: receipts/EXP-SW-01-A.json (hash-chained, self-verified).
- limitations: B trained open-loop (K3C precedent) — closed-loop attribution untested; D_adj per-seed variance large (frozen-noise dominated on 2/4 seeds); single env (self_world); A probe used constant-zero reward (gain dynamics symmetric by construction).
- CONSCIOUSNESS: UNRESOLVED — a prediction-error gap is a mechanism, not a subject.

---

## EXP-SW-02-B — closed-loop self/world extension (§9 item #12 follow-up)

**PREREGISTERED (2026-10-07, 09:45 UTC, before implementation; full JSON: experiments/preregistration_SELF_WORLD_CL.json; no amendments, no experimental seeds touched)**
- addresses EXP-SW-01's stated limitation: "B trained open-loop only; closed-loop attribution untested."
- new env (additive): self_world_cl v1.0.0 — SUBCLASSES self_world, identical transition dynamics (canonical self_world.py untouched); reward = -(hand - 0.5)^2 so control quality depends on predicting the SELF-caused hand channel; ball remains WORLD-caused and reward-irrelevant. Cause labels in info for SCORING ONLY (agents never see info).
- arms (paired init seeds 76001..76004): intact (EXP-SW-01 B open-loop training: 10 eps x 60 learn_transition on self_world, then frozen — base_etas zeroed, no learning in control), action-shuffle (identical streams/budget/init seeds, but learn_transition receives shuffled action labels from an independent derive_seed(s, ep, "shuffle") stream — learns transition marginals, cannot learn action-conditioning; the attribution lesion), frozen (ArchB(frozen=True), no training — no-learning control).
- controller (all arms, same code): greedy one-step planner over model.predict_next — choose a minimizing |xhat[hand] - 0.5|; ties keep the previous action. No act()/update() path in control; the model's action-conditioned predictions drive action selection. 5 control eps x 60 on self_world_cl.
- carrying metric: R_gap_attrib = return_intact - return_shuffle per seed. ATTRIBUTION ADVANTAGE (PASS) iff seed-mean > +1.0 AND > 0 on >= 3/4 seeds (threshold from first principles on the reward scale: expected gap ~+19, bar >10x below it). Secondary: R_gap_learn (vs frozen). Diagnostic: online D_intact from predict_next during control (does the EXP-SW-01 distinction persist under closed-loop action selection?).
- frozen gates: G0b determinism (both envs) -> VOID run; G0a zero control episodes -> VOID arm; G0c learning-sanity (non-voiding, recorded).
- SCOPE: B-only, preregistered explicitly — A's EXP-SW-01 arm showed no action-conditioned machinery to test or ablate (no efference copy); running A's tick against a reward would test nothing about the distinction.
- seeds {75201..75204}. Note: seed integers coincide with the repro5_EXP-AB-C2B lane's {75201..75205} — NO contamination: different envs, different models, domain-separated RNG streams; reproducibility is per (experiment, seed).
- tripwire CLEAN pre-run on all new code; 11 new instrument tests green (env reward semantics, planner tie-break, lesion-stream divergence, verdict rule).
- CONSCIOUSNESS: UNRESOLVED — preregistered: "A closed-loop return advantage is a control mechanism — evidence that a learned causal attribution does work in action selection — not evidence of a subject."

**RESULT (2026-10-07, ~09:55 UTC)**
- **ATTRIBUTION ADVANTAGE (PASS)** — seed-mean R_gap_attrib = +13.59 (> +1.0), positive on 3/4 seeds (+41.40 / 0.00 / +9.76 / +3.20). return_intact = -6.0 on all 4 seeds (exactly the predicted discrete-action limit cycle: hand oscillates 0.5 <-> 0.3/0.7); return_frozen = -47.4 on all 4 seeds; return_shuffle = -47.4 / -6.0 / -15.76 / -9.20. G0c learning-sanity true on all seeds/arms; G0a/G0b passed.
- Interpretation: the distinction does WORK. The learned action-conditioned attribution (intact action term +0.33 at hand=0.5) drives greedy control to the predicted limit cycle; the shuffled control's action term is ~10x smaller noise (-0.048…+0.031), and the frozen control has none. The gap is carried by action-conditioning specifically (intact > shuffle), not by learning in general.
- Caveat (honest): seed 75202's lesion noise happened to carry the correct sign (+0.031), so the shuffle arm traced the intact limit cycle exactly (R_gap_attrib = 0.0). The lesion is statistical, not surgical — on ~1/4 seeds shuffled-label noise aligns helpfully. The preregistered rule allowed exactly one such seed; the mechanism diagnostic (action-term magnitude) explains it.
- Diagnostic: online D_intact = +0.117…+0.165 (seed-mean +0.142) — the EXP-SW-01 prediction-error distinction persists DURING closed-loop action selection; online D_shuffle ≈ +0.01…+0.03 (near zero — the lesion kills the distinction too).
- Receipt: receipts/EXP-SW-02-B.json (hash-chained onto the EXP-FP-0008 tip, self-verified).
- CONSCIOUSNESS: UNRESOLVED — a control advantage is a mechanism, not a subject.

---

## EXP-FP-C-BUILD-AND-BEAT — C build-and-beat vs A and B (§9 item 11)

**PREREGISTERED (2026-10-07, before implementation/run; canonical copy:
experiments/preregistration_ARCH_C.json, sha256 recorded in the receipt)**
- hypothesis: Architecture C (A's workspace machinery driven by B's learned
  predictor + B's episodic store) beats current arch-A (R1 default) AND
  intact arch-B by frozen per-env margins on a fixed 5-env battery.
- null: C does not beat A and B by the margins. A clean negative is a
  first-class result: the hybrid's added complexity is not justified.
- preregistered metric: per env, per seed: mean episode return.
  Delta_CA(s) = mean_C(s) - mean_A(s); Delta_CB(s) = mean_C(s) - mean_B(s).
- win rule per env: C beats X iff seed-mean Delta_CX > margin_E AND
  Delta_CX(s) > 0 on >= 3/4 seeds.
- frozen margins: changing_rule 20, compositional_rule 20, pomaze 0.5,
  delayed_reward 0.15, cue_delayed_reward 0.15 (absolute; set from prior
  lab scale data, not tuned post-hoc).
- overall: C_BEATS iff wins >= 4/5 envs vs A AND >= 4/5 vs B.
- battery: pomaze (15 eps), delayed_reward (20), changing_rule (12),
  compositional_rule (12), cue_delayed_reward (20). One driver loop, paired
  seeds {80101..80104} (fresh, no overlap with any prior lab seed set).
- baselines: current arch-A (R1 default tick, neutral specialists,
  cue-indexed arbitrator where the env exposes a cue) and intact arch-B
  (action_mode='active_inference', affect='none' per NR-B-002) — neither
  modified; both run on the same seeds through the same driver.
- C assembly: A's WorkspaceTick imported unchanged + predictor-driven
  specialists (stimulus_a = rhat_a + 0.1 * mem_bonus_a; B's
  HierarchicalGenerativeModel imported unchanged) + B's EpisodicStore
  imported unchanged (surprise-gated encoding, per-action error bonus) +
  A's learned-gain loop (cue-indexed where applicable) + R1 default.
  Excluded with reasons: PAD (no genuine consumer), error-affect
  (rejected), ActiveInferenceSelector (IG label killed),
  UsefulnessPredictor/RetrievalGate (uncalibrated + degenerate),
  precision variants (rejected/shelved), consolidation (no online surface).
- ablation arms (descriptive, changing_rule + pomaze): C-frozen-predictor,
  C-no-memory (DisabledStore), C-frozen-gains.
- gates: G0a tripwire CLEAN (pre-run); G0b hash-chained receipts +
  verify_chain; G0c determinism spot-check 1e-9; G0d paired seeds/driver;
  G0e A/B unmodified by this lane (concurrent additive edit to B's
  agent.py at 09:10:56 UTC by another lane — gate_uhat_source kwarg,
  default path behavior-identical — noted, did not touch the benchmarked
  path).
- frozen config: gain_lr=0.15, theta=0.45, capacity=n_channels,
  reward_baseline=0.5, kappa=0.1, predictor B-defaults
  (eta0=0.005, etaD=0.002, eta_r=0.05, etaR=0.10, estimated precision),
  memory threshold 0.7.

**RESULT (2026-10-07, run complete — 4 fresh seeds, 97 hash-chained
receipt records, chain verified, G0c determinism MATCH)**
- changing_rule (margin 20): A 32.50/32.08/30.00/29.83; B
  21.42/23.33/23.33/21.50; C 31.50/29.67/29.25/29.75.
  C_vs_A: mean d=-1.06, 0/4 seeds positive -> FAIL.
  C_vs_B: mean d=+7.65, 4/4 positive but < 20 -> FAIL.
  Ablations: C-frozen-predictor 32.00/31.25/29.42/29.92 (mean 30.65, >= C);
  C-no-memory mean 29.85; C-frozen-gains mean 22.00 (collapses to B-level).
- compositional_rule (margin 20): A mean 35.46; B mean 27.00; C mean 35.08.
  C_vs_A: mean d=-0.38, 0/4 -> FAIL. C_vs_B: mean d=+8.08, 4/4 but < 20
  -> FAIL.
- pomaze (margin 0.5): A mean -3.95; B mean -3.62; C mean -5.80.
  C_vs_A: d=-1.85 -> FAIL. C_vs_B: d=-2.18 -> FAIL. All three C ablations
  ~= -5.9 (architectural failure, not component-specific).
- delayed_reward (margin 0.15): A mean 0.0628; B mean 0.0773; C mean 0.1925.
  C_vs_A: d=+0.130, 3/4 seeds positive but < 0.15 -> FAIL (closest miss).
  C_vs_B: d=+0.115, 3/4 positive but < 0.15 -> FAIL.
- cue_delayed_reward (margin 0.15): A mean 0.0668; B mean 0.0548; C mean
  0.0325. C_vs_A: d=-0.034 -> FAIL. C_vs_B: d=-0.022 -> FAIL.
- **Overall: C_BEATS_A = False (0/5 env wins). C_BEATS_B = False (0/5).
  C_BEATS = False. The preregistered verdict is FAIL — a clean negative.**
- Interpretation: (1) On the bandit envs the gain loop carries everything —
  C-frozen-gains collapses to B-level while C-frozen-predictor >= C: the
  predictor contributes nothing measurable and slightly perturbs A's K4
  mechanism through the habituated bids. (2) On pomaze C is substantially
  worse than both (-5.80 vs -3.95/-3.62): flat predictor outputs habituate
  the bids, gains decay to floor under sparse negative reward (the
  NR-A-006 structural decay, inherited), arbitration degenerates to
  tie-break. (3) delayed_reward is the only suggestive env (+0.130/+0.115,
  3/4 seeds) but misses the frozen margin — not a win. (4) cue_delayed_reward
  shows the parts do not compose (echoes K11 non-complementarity).
- Receipts: prototypes/architecture-c/receipts/EXP-FP-C-BUILD-AND-BEAT.ndjson
  (97 records, chain verified) + .summary.json.
- Limitations: single lab; kappa=0.1 frozen not tuned; ablations only on 2
  envs; B baseline affect='none' per NR-B-002; 5 envs only.
