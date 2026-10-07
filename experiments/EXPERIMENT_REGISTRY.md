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
