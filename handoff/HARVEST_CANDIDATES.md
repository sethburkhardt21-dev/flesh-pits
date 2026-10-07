# Harvest classification (directive §49) — Phase 4, 2026-10-07

*Nothing merges into primary automatically. Each HARVEST_NOW candidate below carries source, mechanism, baseline, result, ablation, integration surface, benefit, risks, rollback. Provenance: lab files under ~/workspace/chambers/emergent-mind/ (not git-pinned; paths + dates recorded). Model dependencies: none — all pure Python, stdlib only. License: lab-original code, owner-held.*

## HARVEST_NOW

1. **attention_arbitration + habituation_bid** — Source: sandbox/being/malice_port/ + brothel/prototypes/architecture-a/attention.py, bids.py. Mechanism: deterministic winner-take-all over z-score change+habituation bids. Baseline: fixed salience rules. Result: K4 1.57–2.03× learned/frozen (4/4 seeds; canonical re-run 1.44–1.70). Ablation: freezing gains collapses the advantage (K4). Generalization: bounded — salience-orthogonal relevance shifts only (NR-A-005). Integration surface: any selection/competition point. Benefit: adaptive prioritization with receipts. Risks: M5 caveat — per-channel gains can't do cue-conditional policy; needs cue-indexed adapter. Rollback: revert to fixed salience rules (kept as baseline).
2. **Bounded workspace buffer + sole-path broadcast pattern** — Source: brothel/prototypes/architecture-a/workspace_buffer.py, broadcast.py, tick.py. Mechanism: capacity-K buffer, in-memory bus as the ONLY consumer path. Baseline: unbounded queue / unwired broadcast (Δ=0 theater). Result: K1 interference monotonic at K=3, zero at K=∞; K2 total lesion silences all consumers at once, full recovery, no side channels (reproduced by coordinator). Ablation: K1/K2 are the ablations. Integration surface: any multi-consumer integration point. Benefit: provable global availability. Risks: adds a bottleneck — the burden of proof (K1/K2) must travel with it. Rollback: remove bus, direct calls.
3. **Recurrent ignition gate (bistable threshold)** — Source: brothel/prototypes/architecture-a/ignition.py. Mechanism: iterate-to-convergence amplification, Heaviside gate on propagation. Baseline: linear graded propagation. Result: K3 step width 0.100 < 0.12 preregistered. Ablation: graded control reads linear. Benefit: genuine ignition event, not weighted averaging. Risks: theta=0.6 stationarity perseveration (NR-A-004) — action selection gated on ignition can freeze under full stationarity. Rollback: replace gate with top-k pass-through.
4. **change_bid (RunningZScoreBid)** — Source: sandbox/being/self_model_port/change_bid.py. Mechanism: z-score change detection + habituation. Baseline: raw thresholds. Result: CAUSAL in two independent tests (battery: zero-latency shift detection, drift 22 vs raw 69; A: K1/K4 loop). Benefit: cheap honest novelty signal. Risks: none identified. Rollback: trivial.
5. **consolidation_cycle** — Source: sandbox/being/memory_port/. Mechanism: offline priority consolidation. Baseline: no consolidation. Result: CAUSAL (battery). Benefit: measurable offline improvement. Risks: priority function needs validation per deployment. Rollback: disable consolidation pass.
6. **memory_provenance hash-chaining** — Source: live tree core/memory_port/memory_provenance.py (already live-wired; listed for completeness). Mechanism: hash-chained ProvenanceStore on memory writes. Result: CAUSAL, tick-fresh in live var/being. Benefit: tamper-evident memory writes. Risks: 'agent' origin unreachable via public API (design gap noted). Rollback: n/a — already live.
7. **Experiment harness (preregistration + hash-chained receipts)** — Source: brothel/experiments/harness/ + flesh-pits/experiments/harness.py. Mechanism: preregistered experiments, canonical episode loop, verify_chain() receipts. Baseline: ad-hoc scripts. Result: proven on stubs; 9+8 receipts written this phase. Benefit: every future claim arrives with a receipt. Risks: two harness variants exist (flesh-pits + brothel) — consolidate to one. Rollback: n/a (metrology).

## EXPERIMENT_ON_PRIMARY

- **learned_valence TD estimator** — CAUSAL as an estimator (converges 10.06 ≈ theory 10.26) but never instantiated; decision bite +0.117 < 0.15 bar. Needs: instantiation + decision-bite test before harvest.
- **B's learned predictor core (2-level hierarchy)** — survives K1/K2/C1, beats chance on changing-rule, but hierarchy effect weak and single-seed. Needs: multi-seed, sharper metrics, longer horizons.
- **A's learned-gain attention loop** — needs multi-task generalization (delayed-reward, noisy, multi-reversal) beyond the one task family.

## KEEP_PARALLEL

- B's hierarchy and precision research (precision killed under shift — needs shift-robust variant work, not deployment).
- Full Architecture A prototype (pending multi-seed reproduction).
- (b)-class modules that execute: affective_modulation, coherence_gated_attention, oscillatory_binding — research objects only, NOT tick-path members.
- The 0.5B↔1.5B model-independence test (rung not downloaded).

## REJECT

See REJECTED_IDEAS.md — error-derived affect as controller, "active inference" label (current EFE proxy), unwired broadcast theater, attention_bias (dead), identity-directed affect-remapping guard in pad_emotion_core (§14), scalar-called-phi, "WIRED (enrichment only)" read as causal.

## INCONCLUSIVE

- Architecture C (unbuilt by design).
- Tag-at-encoding via replay/PE; PE-gated retrieval update (unimplemented).
- 'agent' provenance genesis path (design gap).
- All multi-seed reproduction (Phase-4 experiments are single-seed).
