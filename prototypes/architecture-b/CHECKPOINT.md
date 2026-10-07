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

## Open / BLOCKED
- Single seeds everywhere — nothing REPRODUCED. Multi-seed rerun needed before harvest.
- No-memory ablation not run (EpisodicStore causal status unknown).
- Retrieval-usefulness prediction not wired to gate retrieval.
- C2/C4 need re-preregistered sharper metrics (shift-robust precision; uncertainty-reduction IG).
