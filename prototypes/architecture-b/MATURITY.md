# CAUSAL MATURITY — Architecture B prototype v1

*Directive §41. Tracked honestly. 2026-10-07.*

Scale: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.

## Component maturity

| Component | Maturity | Evidence |
|---|---|---|
| HierarchicalGenerativeModel (L0 linear + L1 context experts; persistent mu0) | **ADAPTIVE** | K1 survive (freeze → -4.6 return); K2 survive (learned err 0.98 < fixed 1.32); BAR intact 24.4 > lesioned 21.4 > chance 20.0; R table re-learns after flip. NOT generalizing (contingency only shown on changing_rule). NOT reproduced (single seeds). |
| PrecisionEstimator (pi from error stats) | **CAUSAL, REJECTED** | C2 KILL (replicated 2 seeds): uniform pi=1 beats estimated pi by 5.8–9.5 return points on changing_rule. Causally load-bearing but HARMFUL under distribution shift (overshoot after flips). Do not ship as-is. |
| EpisodicStore (experience_store port + provenance) | **INTEGRATED** | Stored every tick; retrieval correction applied to xhat; per-action errors feed IG. CAUSAL not established — no-memory ablation not run (gap). |
| §30 next-obs / action-consequence predictions | **INTEGRATED** | Logged every tick (JSONL); consumed by action selection. Calibration tracked. |
| §30 retrieval-usefulness prediction | **EXECUTED** | Predicted + measured every tick, but uhat does NOT gate retrieval (unconditional correction) → not integrated. Gap. |
| ActiveInferenceSelector | **INTEGRATED, label removed** | C4 KILL: greedy beat AI on preregistered IG metric (0.0502 > 0.0275). AI had best return (-3.8), but the IG term's causal contribution is unproven. The "active inference" label comes off per the kill condition. |
| ErrorAffect (error-derived valence/arousal) | **CAUSAL, REJECTED** | K4 KILL: PAD (-0.762) beat error-affect (-0.787) on resource_world. NR-B-002: arousal saturates → permanent over-exploration. Dropped; PAD kept as baseline. |
| PadController (donor-exact baseline) | **CAUSAL** | Won K4. Kept as the cheaper baseline per the kill rule. |

## Kill-experiment outcomes

| ID | Result | Verdict |
|---|---|---|
| K1 learning freeze | learn 26.2 > frozen 21.6 (+4.6) | SURVIVES — not a static function |
| K2 learned vs fixed | learned err 0.98 < fixed 1.32 | SURVIVES — learning not decorative |
| K3 hierarchy lesion | struct +0.0032 (lesioned worse), white-noise -0.0012 (flat) | SURVIVES, WEAK — effect tiny; metric diluted by unpredictable channels |
| K4 affect vs PAD | PAD -0.762 > error -0.787 | KILL — drop error-affect, keep PAD |
| K5 shuffle | shuffled 0.8606 > ordered 0.8575 | SURVIVES, WEAK — gap tiny; same dilution |
| C1 error decline | delayed_reward ✓, grid_world ✓ | HOLDS |
| C2 precision ablation | uniform wins by 5.8–9.5 (2 seeds) | KILL — precision harmful under shift |
| C4 active inference | greedy IG > AI IG | KILL — label comes off |
| BAR concrete bar | intact 24.4 > lesioned 21.4 > 20.0 chance | MIXED — prereg decline metric confounded by flip; within-phase + return evidence supports hierarchy |

## Open gaps (honest)

1. Single seeds everywhere — nothing is REPRODUCED.
2. No-memory ablation (EpisodicStore causal status unknown).
3. Retrieval-usefulness prediction not wired to gate retrieval.
4. Hierarchy generalization beyond changing_rule untested.
5. C2/C4 metric flaws identified — re-preregister sharper metrics before
   any harvest claim.

CONSCIOUSNESS: UNRESOLVED — mechanisms, not narratives.
