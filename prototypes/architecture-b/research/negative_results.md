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
