# Negative results — Flesh Pits (§40, §52)

Negative results are legitimate success when they eliminate bad hypotheses.
Each entry: experiment ID, what was expected, what happened, what it rules
out. Preregistrations live in experiments/EXPERIMENT_REGISTRY.md; per-run
evidence in receipts/.

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
