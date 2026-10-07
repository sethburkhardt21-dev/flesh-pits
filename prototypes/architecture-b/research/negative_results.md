# Negative results — Architecture B (2026-10-07)

*Reconstruction note (maturity audit, 2026-10-07): CHECKPOINT.md claims
NR-B-001..006 were "appended to research/negative_results.md" — no such
file ever existed at the architecture-b level or the lane level, so the
original entries are lost. Entries below are reconstructed from surviving
sources (ARCHITECTURE_B.md design decisions, CHECKPOINT.md, MATURITY.md,
handoff/REJECTED_IDEAS.md, receipts). NR-B-003..006 have no surviving
description and are recorded as LOST, not invented.*

## 2026-10-07 — NR-B-001: mixture-of-experts L1 cold-start (redesign)

- Expectation: a mixture-of-experts L1 with input-free gating (mu1 vector)
  would learn cue×action contingencies.
- Observed: fatal cold-start — gating weights and experts needed each other
  for gradient; reward predictions stayed flat across actions after 200
  trials.
- Resolution: replaced with context-indexed experts (v2): the active
  context's expert updates directly — clean credit assignment, no
  chicken-and-egg. The failure is preserved as the reason v2 exists.
- Source: ARCHITECTURE_B.md design decision #1;
  handoff/REJECTED_IDEAS.md entry 10b.

## 2026-10-07 — NR-B-002: error-derived affect saturates arousal

- Expectation: valence/arousal derived from error dynamics would beat a PAD
  dictionary as a controller.
- Observed: K4 — PAD −0.762 beat error-affect −0.787 on resource_world.
  Mechanism: error-derived arousal saturates at 1.0 on volatile tasks →
  explore_gain=2.0 permanently → the agent explores forever and sits at
  chance (20.0 vs 26.0 with affect off on changing_rule). World-model kill
  experiments run with affect="none"; the affect controller is judged only
  by K4.
- Rules out: "error-derived affect as a controller." Dropped; PAD kept as
  the cheaper baseline per the kill rule.
- Receipt: receipts/EXP-AB-K4.json. Source: ARCHITECTURE_B.md design
  decision #4; MATURITY.md ErrorAffect row; handoff/REJECTED_IDEAS.md
  entry 1b.

## 2026-10-07 — NR-B-003..006: LOST

Claimed by CHECKPOINT.md ("appended to research/negative_results.md",
NR-B-001..006) but no file was ever written and no surviving source
describes NR-B-003, NR-B-004, NR-B-005, or NR-B-006. Recorded as lost;
not reconstructed — inventing entries would be fabrication.

## 2026-10-07 — NR-B-007: shift-robust precision variant killed (C2B)

- Expectation: the ONE preregistered shift-aware precision variant
  (surprise-triggered window reset, k=4.0) would survive distribution
  shift where estimated precision failed.
- Observed (3 seeds): shift arm 23.97 (shift_reset) < 24.10 (estimated) <
  28.47 (uniform); stationary arm 24.40 < 34.00 (uniform).
- Diagnosed: max-channel surprise detector fired 82–110×/run (expected
  ~1 — cannot separate rule flips from single noisy trials), and each
  post-reset cold start reintroduces the overshoot via tiny-window pi
  estimates.
- Rules out: precision weighting in both tested forms (estimated,
  shift-reset). The idea is shelved, not merely paused.
- Receipt: receipts/EXP-AB-C2B.json. Source: CHECKPOINT.md Phase-4;
  handoff/REJECTED_IDEAS.md entry 3b.

## 2026-10-07 — NR-B-008: "active inference" label killed permanently (C4B)

- Expectation: the ActiveInferenceSelector would beat greedy/random on a
  sharper, policy-independent uncertainty-reduction metric (probe-set
  prediction-error decline, identical init).
- Observed (3 seeds): AI 0.5443 < greedy 0.7124 ≈ random 0.5497 — kill
  confirmed permanently. The IG term's causal contribution is unproven
  under both metrics.
- Caveat (recorded, verdict unchanged): the probe metric rewards
  concentrated practice (greedy's narrow-deep experience) — a lesson for
  future IG metric design, not a verdict change.
- Receipt: receipts/EXP-AB-C4B.json. Source: CHECKPOINT.md Phase-4;
  handoff/REJECTED_IDEAS.md entry 2b.

## 2026-10-07 — K3C: hierarchy does not generalize beyond the two K3B envs (hierarchy-scope bound)

- Expectation (preregistered; full spec
  flesh-pits/experiments/preregistration_EXP-AB-K3C.json): B's L1
  hierarchy would earn its keep on harder tasks — intact-vs-L1-lesioned
  |e0| gap positive on delayed_multistep (3× horizon, two-stage corridor
  with junction; scripted reference policy, 30 episodes, train eps 0–19,
  held-out eps 20–29), and |rerr| gap positive on compositional_rule
  (cue_a×cue_b×action XOR contingency, single-cue marginals exactly
  50/50; 45 episodes, train eps 0–39, held-out eps 40–44). Gates: seed-mean
  carrying gap > +0.0032 (K3's weak-effect scale) AND positive on ≥3/4
  fresh seeds {74101–74104}; both pass → SCOPE HOLDS, one → SCOPE BOUNDED,
  neither → NO EFFECT.
- Observed (4/4 seeds, K3B-identical open-loop instrument):
  task 1 D_e0 seed-mean −0.0470 (−0.036…−0.059, negative 4/4) — disabling
  L1 IMPROVES dynamics prediction on the longer multi-stage horizon;
  task 2 D_rerr seed-mean −0.2684 (−0.217…−0.341, negative 4/4), D_e0
  −0.0097 (negative 4/4) — the lesion helps on both channels. Verdict:
  **NO EFFECT — the hierarchy is a two-env phenomenon** (changing_rule /
  delayed_reward only).
- Diagnosed (post-hoc, not preregistered): on phase-7 held-out (the phase
  the context tables trained on) intact|rerr|=0.22 vs lesioned 0.51
  (D=+0.29 — the tables DO capture the XOR within a stable phase); on
  phase-8 held-out (flipped maps) intact 0.79 vs lesioned 0.57. The R
  tables memorize the old phase's contingency and predict it confidently
  after the flip, while L0-only degrades gracefully to chance.
  Context-specificity is a liability under distribution shift — the same
  cross-cutting pattern as the Phase-3 replications (K1/K3/C2: L1-context
  and precision machinery does not robustly convert learning into better
  decisions under shift).
- Methods note: the preregistered task-1 expectation "terminal |rerr| ≈ 0
  by construction" was corrected pre-run by amendment (throwaway-seed
  probe): the intact model's context tables learn per-context base rates
  while the lesioned model's estimated-precision reward head explodes on
  the rare terminal spikes (verified: uniform precision keeps it sane,
  0.10 vs 0.83) — same precision pathology family as the C2 kill. The
  reward channel on task 1 was kept as a diagnostic, not the carrying
  metric; the verdict rests on the preregistered D_e0/D_rerr gates.
- Rules out: "the hierarchy earns its keep on harder tasks." Extends the
  hierarchy-scope question (cf. NR-B-006, recorded LOST above — not
  reconstructed; K3B's two-env evidence stands separately).
- Receipts: receipts/EXP-AB-K3C.json (arch-b detail, hash-chained to
  EXP-AB-K3B); ../../receipts/EXP-AB-K3C.json (lane summary, harness
  chain). New envs: flesh-pits/experiments/envs/delayed_multistep.py,
  compositional_rule.py (registered additively; canonical envs untouched).

## 2026-10-07 — NR-B-009: K3B's reward-channel hierarchy signal reverses on fresh seeds (weakening, not kill)
- Expectation: the K3B verdict (SURVIVES, STRENGTHENED) would replicate on 5 fresh seeds — arm-A reward-channel lesion gap staying positive at ~19× K3's +0.0032 scale (+0.0602, 3/3 seeds).
- Observed (repro5_EXP-AB-K3B, 5 fresh seeds {75401..75405}, identical instrument, preregistered): the frozen gate fires REPRODUCES at exactly 4/5 seeds (per-seed "grown" = delta_e0 > 0.0032 OR delta_rerr > 0.0032 on arm A). But the carrying channel REVERSED: arm-A |rerr| gap −0.0319 seed-mean, negative on 3/5 seeds (75402 −0.0305, 75403 −0.0855, 75404 −0.2753 — the lesion HELPS reward prediction there). The replication's signal rides the |e0| channel (+0.0079 seed-mean, positive 4/5), at only ~2.5× K3's scale. Arm B: |e0| +0.0356 (reproduces direction), terminal |rerr| −0.0224 (≈0, as preregistered).
- Rules out: "K3B's +0.0602 reward-channel gap is a robust property of the hierarchy on longer horizons." The +0.0602 was seed-fragile. The hierarchy's longer-horizon contribution is, at best, a thin dynamics-channel effect; its locus and magnitude are not stable across seeds.
- Verdict note: REPRODUCES per the frozen gate (4/5) — this is a substantive weakening, not a verdict change. The two-env phenomenon (K3C) stands; the scope bound tightens further.
- Receipts: receipts/repro5_EXP-AB-K3B.json (arch-b detail, chain-linked to EXP-AB-K3B); ../../receipts/repro5_EXP-AB-K3B.json (lane summary, harness chain).
