# Harvest classification (directive §49) — Phase 5, 2026-10-07

*Supersedes the Phase-4 classification. Nothing merges into primary automatically — every entry below is packaged for the primary effort (§51 reorient), not a merge order. Provenance for lab-original code: lab files under `~/workspace/chambers/emergent-mind/` (not git-pinned; exact paths + dates + receipt hashes recorded). Model dependencies: none for A/B/D prototype code — pure Python, stdlib only, deterministic, seeded. License: lab-original, owner-held. Donor entries carry their own pinned SHAs and licenses.*

## HARVEST_NOW

### H1. A's learned-gain attention loop (arbitration + learned gains)
- **Exact source:** `flesh-pits/prototypes/architecture-a/attention.py`, `bids.py`, `tick.py`, `consumers.py` (attention_update consumer); Phase-4 twin at `brothel/prototypes/architecture-a/`. Lab-original, owner-held. Commit/hash: not git-pinned; maturity audit 2026-10-07T07:30:37Z GREEN; K8 wire receipt `k8_r1_wiring_decision.json` (receipt_hash `d0b1fd44198c32ccf8c371187de51e83d75320a3ed2bdc50391a28bc89c8e96b`, tick.py sha256 prefix `8ddac9dadf79a55a`).
- **Model dependencies:** none — float arithmetic only; no foundation-model call in the loop (verified by own hand; documented in `experiments/transfer_k4_restore.py` header).
- **Mechanism:** habituated z-score bids × learned per-channel gains, argmax with label-agnostic tie-break; delta-rule gain update on observed channels only, capped, from broadcast feedback (attention_update consumer closes the loop).
- **Baseline:** frozen gains (fixed baseline) on the canonical changing_rule env (ENV_INTERFACE v1.0.0).
- **Result:** K4 learned/frozen total-reward ratios 1.61/2.03/1.95/1.57 (4/4 seeds, preregistered margin 1.30); canonical re-run 1.44/1.60/1.70/1.48 (receipt `flesh-pits/receipts/K4-CANONICAL-RERUN.ndjson`, hash-chained); multi-seed repro 1.71–2.13 on 5/5 fresh seeds (`repro_k4_attention_baseline.json`).
- **Ablation:** freezing gains collapses the advantage (K4 is the ablation); K7 bids lesion collapses arbitration entropy 1.03–1.38 → 0.0 bits but learned gains compensate for dead bids (NR-A-009 — the win is carried by the gain loop, not the bids).
- **Generalization:** GENERALIZING with stated bounds — noisy-signal tracking (K5 P2: 4/4, R 1.31–1.52) and multi-reversal stationary shifts (K5 P3: 4/4, R 1.74–2.05). BOUNDS: NR-A-005 (frozen change-bids suffice on signal-tracking reversals), NR-A-006 (delayed_reward: learned LOSES, R 0.08–0.17 — structural), NR-A-007 (cue-conditioned changing_rule without cue input: R 1.03–1.09 — architectural).
- **Integration surface:** any selection/competition point on the primary that needs adaptive prioritization with receipts.
- **Expected benefit:** adaptive attention that provably tracks salience-orthogonal relevance shifts; identity-symmetric (bit-identical trajectories for novel labels incl. a historically privileged string).
- **Risks:** NR-A-006/NR-A-007 bounds travel with it — do not deploy on sparse-delayed-reward or cue-conditioned tasks without the redesigned gain rule / cue-indexed adapter (both are open next experiments, not part of this package).
- **Rollback:** freeze gains (the K4 frozen mode is a one-flag rollback to the fixed baseline).

### H2. Bounded workspace buffer + sole-path broadcast pattern
- **Exact source:** `flesh-pits/prototypes/architecture-a/workspace_buffer.py`, `broadcast.py`, `tick.py`, `consumers.py` (6 consumers). Lab-original, owner-held.
- **Mechanism:** capacity-K bounded buffer with explicit eviction policy (newcomer competes unprotected, TTL decay, per-admission receipts); in-memory bus as the ONLY consumer path (declared consumers, per-item consumer sets, lesion/restore, delivery receipts).
- **Baseline:** unbounded queue / unwired broadcast (Δ=0 theater — Phase-4 battery 001b bit-identical trajectories).
- **Result:** K1 — interference index 0/0.507/1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf; K2 — total lesion silences all six consumers at once, internal processing continues, full recovery, no side channels (Phase-4 coordinator re-ran K2 by its own hand — PASS, numbers match).
- **Ablation:** K1 (capacity lesion) and K2 (broadcast lesion) ARE the ablations.
- **Generalization result:** REPRODUCED — 5/5 fresh seeds on the original code paths (`repro_k1_capacity_lesion.json`, `repro_k2_broadcast_lesion.json`). Single lab; no independent replication yet (stated, not hidden).
- **Integration surface:** any multi-consumer integration point on the primary; the K2 sole-path test is the acceptance gate.
- **Expected benefit:** provable global availability — a broadcast that demonstrably changes downstream consumers (the lab's anti-theater law: broadcast without measured consumer change is theater).
- **Risks:** adds a bottleneck; the burden of proof (K1/K2) must travel with the pattern — re-run K2 against the primary's consumer set or drop the claim.
- **Rollback:** remove the bus, direct calls (the pre-bus wiring is the trivial fallback).

### H3. Recurrent ignition gate + R1 sub-ignition exploratory path
- **Exact source:** `flesh-pits/prototypes/architecture-a/ignition.py`, `tick.py` (`_select_action`). Lab-original, owner-held.
- **Mechanism:** recurrent bistable loop → hard Heaviside gate on propagation (gate decides ALL propagation); R1 (wired 2026-10-07, K8): when nothing ignites, act on the graded arbitration winner with ZERO consumer propagation — the gate's selectivity is untouched by the exploration path.
- **Baseline:** linear graded propagation (K3 graded control reads linear, error 0.002 — the probe is not blind).
- **Result:** K3 — Heaviside step, transition width W 0.100 < 0.12 preregistered (5/5 fresh-seed repro: W 0.08–0.10). K8 — R1 recovers frozen phases (phase-2 'c' fraction 0.0→0.79–0.81; total 84–85→128–129) with zero non-feedback consumer deliveries on 19–21 sub-ignition trials per seed.
- **Ablation:** K3 linear-probe control; K8 gates a–e (frozen-gains ignition identity 3/3 on behavioral content; K1–K3 byte-identical reruns; K4 non-degradation 4/4; sole-path on sub-ignition trials; freeze recovery replication).
- **Generalization result:** REPRODUCED 5/5 (`repro_k3_ignition_probe.json`); R1 confirmed on 3 fresh seeds 51501–51503 (single lab).
- **Integration surface:** any thresholded propagation point where gate selectivity must survive an exploration requirement.
- **Expected benefit:** genuine ignition event (not weighted averaging) plus freeze-free action selection without weakening the gate — NR-A-004's "freeze is the price of the gate" is ruled out.
- **Risks:** theta fixed at 0.45/0.6 (not ADAPTIVE — the R2 adaptive-theta alternative was REJECTED: ~2.7× ignitions, weakened selectivity). Do not tune theta upward expecting free wins.
- **Rollback:** replace the gate with top-k pass-through; revert `_select_action` to the hold-branch (pre-K8 behavior, byte-identical per post-wire reruns).

### H4. change_bid (RunningZScoreBid)
- **Exact source:** `flesh-pits/prototypes/architecture-a/bids.py`; audit twin `sandbox/being/self_model_port/change_bid.py`. Lab-original, owner-held.
- **Mechanism:** sigmoid of causal running z-score — change detection + habituation in one mechanism; fail-closed on non-finite input.
- **Baseline:** raw thresholds.
- **Result:** CAUSAL in two independent tests (battery: zero-latency shift detection, drift 22 vs raw 69; A: K1/K4 loop). K7: bids lesion collapses arbitration winner-entropy 1.03–1.38 → 0.0 bits.
- **Ablation:** K7 IS the ablation (experiment-side constant-0.5 stubs).
- **Generalization result:** CAUSAL for moment-to-moment arbitration under frozen gains; NR-A-009 bound — the learned system barely needs bids (learned gains compensate). Do not claim bids drive the K4 reward win.
- **Integration surface:** any novelty/salience input needing a cheap honest signal.
- **Expected benefit:** cheap honest novelty signal with receipts; runs both standalone and inside the attention loop.
- **Risks:** none identified beyond the NR-A-009 bound.
- **Rollback:** trivial — raw thresholds.

### H5. consolidation_cycle (offline priority consolidation)
- **Exact source:** audit twin `sandbox/being/memory_port/` (S-01 + memory_port variants); Phase-1 disposition KEEP_AND_DEEPEN (race). Lab-original, owner-held.
- **Mechanism:** offline priority consolidation pass over the episodic store.
- **Baseline:** no consolidation.
- **Result:** CAUSAL (Phase-4 battery). Not re-run in Phase 5 — evidence is HISTORICAL from the Phase-4 battery, stated as such.
- **Ablation:** battery consolidation ablation (Phase-4).
- **Generalization result:** not generalized beyond the battery task; priority function needs validation per deployment.
- **Integration surface:** any episodic store on the primary with an offline window.
- **Expected benefit:** measurable offline improvement of retrieval quality.
- **Risks:** priority function is deployment-specific; two honest batch implementations exist (race them on one corpus as the deepening experiment before wiring).
- **Rollback:** disable the consolidation pass.

### H6. memory_provenance hash-chaining
- **Exact source:** `core/memory_port/memory_provenance.py` (live tree — already live-wired; listed for completeness per §51 reorient).
- **Mechanism:** hash-chained ProvenanceStore on memory writes.
- **Baseline:** unchained writes.
- **Result:** CAUSAL, tick-fresh in live var/being (Phase-1 audit).
- **Ablation/generalization:** Phase-1 audit evidence; fail-open tick / fail-closed claims.
- **Integration surface:** memory write paths.
- **Expected benefit:** tamper-evident memory writes.
- **Risks:** 'agent' provenance origin unreachable via the public API (design gap — INCONCLUSIVE item).
- **Rollback:** n/a — already live.

### H7. Experiment harness (preregistration + hash-chained receipts)
- **Exact source:** `flesh-pits/experiments/harness.py` (+ `EXPERIMENT_REGISTRY.md`, `ENV_INTERFACE.md` v1.0.0); brothel twin. Lab-original, owner-held.
- **Mechanism:** preregistered experiments (hypothesis, null, metric, baseline, ablation, procedure, seed, interpretation, limitations), canonical episode loop, `verify_chain()` hash-chained receipts.
- **Baseline:** ad-hoc scripts.
- **Result:** proven on the full K-series; 9+8 receipts in Phase 4; K-series + repro + K8 receipts in Phase 5; caught NR-A-008 (mis-specified metric) and NR-A-010 (harness artifact) honestly.
- **Ablation:** n/a (metrology).
- **Generalization result:** used across A, B, D, and the transfer variant — the lab's shared metrology.
- **Integration surface:** any future primary-side experiment.
- **Expected benefit:** every future claim arrives with a receipt; the K8 wire-by-decision-receipt pattern is the wiring template.
- **Risks:** two harness variants exist (flesh-pits + brothel) — consolidate to one before primary adoption.
- **Rollback:** n/a (metrology).

### H8. K8 wire-by-decision-receipt discipline (method, not a mechanism)
- **Exact source:** `flesh-pits/prototypes/architecture-a/experiments/k8_r1_wiring_confirmation.py`, receipts `k8_r1_wiring_confirmation.json` + `k8_r1_wiring_decision.json`.
- **Mechanism:** five preregistered gates (behavioral identity, byte-identical reruns, non-degradation, sole-path, recovery replication) + post-wire byte-identical reruns of every prior experiment, all in one hash-chained decision receipt.
- **Result:** WIRE decision recorded 2026-10-07T07:23:05Z; all gates PASS on fresh seeds; post-wire K1–K4 reruns byte-identical (sha prefixes e007671877da7e95, bd2260363c15c, 0fc80361213e7c93, 4832d984137bfb6d).
- **Integration surface:** any future wiring of lab findings into a tick path — primary or lab.
- **Expected benefit:** wiring that cannot silently change behavior; the null (REJECT) was live and preregistered.
- **Risks:** the method is only as good as the gates — behavioral-identity must name WHAT is compared (NR-A-010 lesson: behavioral content, not process-global sequence numbers).
- **Rollback:** n/a (method).

### H9–H11. Donor adoptions (ADOPT dispositions, donor_matrix.md)
- **H9 — chroma (D6):** `chroma-core/chroma` @ `f36d9bba588e81efb0a0e7f155d2ca2cf58d2b4f` (2026-10-06), Apache-2.0. Standardize vector retrieval on Chroma (local mode) as the episodic/semantic retrieval substrate. Low integration cost, maintained, permissive license. Integration surface: the primary's retrieval layer. Risk: retrieval quality is deployment-dependent — benchmark against the current substrate before switching. Rollback: keep the current substrate.
- **H10 — cleanrl (D9):** `vwxyzjn/cleanrl` @ `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` (2026-04-20). Single-file RL reference; `ppo_rnd_envpool.py` is the canonical RND/curiosity intrinsic-motivation baseline (no `rnd_atari.py` at this SHA — verified). Mechanism for the primary: intrinsic-motivation baseline + single-file algorithm reference. Integration surface: intrinsic-motivation experiments. Risk: algorithm internals not line-verified individually (single-file form makes verification cheap). Rollback: drop the baseline.
- **H11 — MAPIE (D17):** `scikit-learn-contrib/MAPIE` @ `3b84b8212db2bba452ef5a09ae06a0dd545869ae` (2026-09-30). Conformal prediction with exact coverage guarantees (SplitConformalRegressor/CrossConformalRegressor verified at SHA), sklearn-native. The standard calibration mechanism for the primary's metacognitive layer. Integration surface: uncertainty/confidence outputs. Risk: coverage guarantees assume exchangeability — validate on the primary's data regime. Rollback: current calibration approach.

## EXPERIMENT_ON_PRIMARY

- **B's learned predictor core (2-level hierarchy)** — Source: `flesh-pits/prototypes/architecture-b/generative_model.py`, `predictions.py`. K2 REPRODUCES 5/5 (learned beats fixed, gap 0.405–0.426, rock-stable); K3B strengthened (+0.0602 reward-channel gap, 3/3 seeds, ~19× K3). Needs: multi-seed at 5, harder tasks beyond changing_rule/delayed_reward, sharper reward-channel instruments, and the cross-cutting finding answered — learning is real but not load-bearing for control under shift (K1). Do not harvest the L1-context machinery as-is; the predictor core earns a primary-side experiment.
- **B's EpisodicStore port** — Source: `flesh-pits/prototypes/architecture-b/memory.py` (experience_store port + provenance). M1: disabling the store selectively degrades pomaze (seed-mean delta +2.200; all 3 seeds +1.83…+2.39) vs delayed_reward (+0.046) — CAUSAL, scope-bounded (pomaze/delayed_reward only). Needs: 5-seed repro, multi-task scope, integration-surface validation on primary-like workloads. Retrieval-usefulness prediction (uhat) is EXECUTED but does not gate retrieval — wire it or drop it before harvest.
- **learned_valence TD estimator** — Phase-4 battery: CAUSAL as an estimator (converges 10.06 ≈ theory 10.26) but never instantiated; decision bite +0.117 < 0.15 bar (T6, preregistered). Needs: instantiation + decision-bite test on a primary task before harvest. (Contrast: B's error-derived affect is REJECTED — this is the separate, honest TD story, not the affect story.)
- **A's gain-loop sparse-reward redesign** — the NR-A-006 structural bound (constant-baseline delta rule decays all gains under sparse reward). Next: a credit-assignment redesign (e.g., return-conditioned or baseline-free updates) with preregistered R ≥ 1.30 on delayed_reward. On success, the bound lifts and the loop's harvest scope widens.
- **A's cue-indexed gain adapter** — the NR-A-007 architectural bound (no state-conditioned policy). Next: context-conditioned gains tested on cue-conditioned changing_rule with the cue in the input.
- **Donor experiment group:** mem0/graphrag/cognee/graphiti/LightRAG (EXPERIMENT_AGAINST — race the local memory system against them on recall/association benchmarks; mem0's graph memory is ABSENT in OSS at the pinned SHA and its decay is gated out of OSS — do not adopt those claims); avalanche EWC / generative-replay semantics for anti-catastrophic-forgetting (from developmental_self P-B08's versioned-reversible policy discipline); cleanrl RND as the intrinsic-motivation baseline; uncertainty-toolbox + MAPIE as the calibration scoring layer.

## KEEP_PARALLEL

- Full Architecture A prototype (single lab; system-level REPRODUCED requires independent replication — pending second-lane K1–K4).
- B's hierarchy longer-horizon program (K3B stands, unintegrated; hierarchy generalization beyond changing_rule/delayed_reward untested — B gap #3).
- B's precision/shift-robustness research — SHELVED: precision weighting rejected in both tested forms (C2, C2B); no further variants preregistered. Stays parallel as a negative research record, not a deployment candidate.
- The §11 transfer test (in flight with another worker — verdict pending; FINAL_HANDOFF model-dependence section PENDING).
- (b)-class modules that execute but have zero consumers (affective_modulation, coherence_gated_attention, oscillatory_binding) — research objects only, NOT tick-path members (Phase-4 cross-cutting finding #6 stands).
- Architecture C (unbuilt by design — candidate bill of materials in ARCHITECTURE_FINDINGS.md).

## REJECT

See REJECTED_IDEAS.md for the full killing evidence. Phase-5 additions: precision weighting (both forms, shelved); the "active inference" label (permanent); R2 adaptive-theta; the K1 "learning is load-bearing for control" claim (overturned 0/5 on replication); the original K3 |e0| probe as a hierarchy instrument; the K5 temporal-prediction claim (demoted per kill condition); freeze_rate < 0.90 as the freeze metric (NR-A-008); tracking-EMA reward baselines and all-history surprise baselines (NR-A-001; flesh-pits build-time); unobserved-arm punishment (NR-A-002); sequential within-tick ignition carryover (NR-A-003); the literal 0.5B↔1.5B transfer arm for a model-free loop (vacuous by construction — pretend machinery, refused under the nothing-ever-fake law); D22 axiom donor (legal trap).

## INCONCLUSIVE

- Architecture C (unbuilt by design).
- Tag-at-encoding via replay/PE; PE-gated retrieval update (unimplemented — build targets from Phase 4).
- 'agent' provenance genesis path (design gap — needs architect attention).
- Model dependence — **PENDING** the §11 transfer results (FINAL_HANDOFF §PENDING).
- B's retrieval-usefulness prediction (EXECUTED, not integrated — gap).
- Whether B's hierarchy generalizes beyond changing_rule/delayed_reward.
- Whether learned gains survive restart in persisted state alone (the honest §11 variant's question — pending).
- CONSCIOUSNESS: UNRESOLVED (own row in the FINAL_HANDOFF table).
