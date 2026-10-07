# CHECKPOINT — Architecture B implementer (2026-10-07)

## Done
- [x] Read architecture_candidates.md (Arch B), ENV_INTERFACE.md, all 5 envs, experience_store + PAD references (read-only).
- [x] Implemented: precision.py, generative_model.py (v2 context experts), memory.py, predictions.py, active_inference.py, affect.py, agent.py — pure Python, stdlib, seeded, deterministic.
- [x] 21/21 unit tests pass (determinism, contract, lesions, neutrality scan, snapshot/restore).
- [x] Ran full battery: K1–K5, C1, C2, C4, BAR. 9 receipts in receipts/. Per-tick §30 log in experiments_out/.
- [x] MATURITY.md (honest per-component levels). Negative results appended to research/negative_results.md (NR-B-001..006).
- [x] Forbidden paths untouched (mneumora mtime 2026-10-06; sleeping-quarters never modified).

## Key outcomes
- SURVIVE: K1 (learning real), K2 (learning not decorative), K3 (weak), K5 (weak), C1 (error declines).
- KILL: C2 (precision harmful under shift, replicated), C4 ("active inference" label off), K4 (PAD beats error-affect).
- BAR: MIXED — intact 24.4 > lesioned 21.4 > chance 20.0, but prereg decline metric confounded by flips.
- Two redesigns during build: MoE→context experts (NR-B-001), mu0 Kalman blend (numerical stability).

## Phase 4 gap-closure (2026-10-07, COMPLETE)

Code additions (all in prototypes/architecture-b/):
- `memory.py`: `DisabledStore` null-object (M1 no-memory ablation).
- `shift_precision.py`: NEW `ShiftRobustPrecisionEstimator` — surprise-triggered
  window reset (k=4.0, preregistered), ONE variant for C2B.
- `generative_model.py`: `precision_kind` ("estimated"|"uniform"|"shift_reset")
  + `surprise_k` wired into prec0/prec1/precR; snapshot/restore extended.
- `agent.py`: `memory_enabled`, `precision_kind`, `surprise_k` knobs;
  `uniform_precision=True` still takes precedence.
- `experiments_phase4.py`: M1, C2B, C4B, K3B. 3 fresh seeds each
  (301-303, 311-313, 321-323, 331-333).
- Smoke tests pass; unit tests 21/21 after wiring changes.

Results (receipts in receipts/EXP-AB-{M1,C2B,C4B,K3B}.json):
- M1: SELECTIVE DEFICIT — pomaze +2.200 (3/3 seeds +1.83…+2.39),
  delayed_reward +0.046. Memory CAUSAL on partial-observability task, not
  decorative. EpisodicStore INTEGRATED → CAUSAL.
- C2B: REJECTED — shift_reset 23.97 < estimated 24.10 < uniform 28.47
  (shift); 24.40 < 34.00 (stationary). Detector fired 82–110×/run
  (expected ~1): max-channel surprise can't separate flips from noise;
  post-reset cold-start reintroduces overshoot. Precision rejected in both
  forms; idea shelved. NR-B-007.
- C4B: KILL CONFIRMED PERMANENTLY — AI probe-IG 0.5443 < greedy 0.7124 ≈
  random 0.5497. Label stays off. (Caveat: probe metric rewards concentrated
  practice — recorded, verdict unchanged.) NR-B-008.
- K3B: HIERARCHY EARNS ITS KEEP — arm-A reward-channel lesion gap +0.0602
  (3/3 seeds, ~19× K3's +0.0032); |e0| gap +0.0054 grew but seed-unstable
  (1/3 reversed). Arm-B |e0| gap +0.0437 consistent; terminal |rerr| ≈ 0 as
  preregistered (hidden branch; metric underpowered, 0–1 terminal ticks).
- MATURITY.md updated (component levels, kill table, open gaps).
  Negative results appended (NR-B-007, NR-B-008).
- Forbidden paths untouched: mneumora/, sleeping-quarters/, brothel/ never
  modified (verify: no writes outside prototypes/architecture-b/ and
  emergent-mind/research/negative_results.md).

## Still open (post-Phase-4)
- Phase-3 battery (K1-K5, C1, C2, C4, BAR) remains single-seed; Phase-4
  experiments run 3 seeds each.
- Retrieval-usefulness prediction still not wired to gate retrieval.
