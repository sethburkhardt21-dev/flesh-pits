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
