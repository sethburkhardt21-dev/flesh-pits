# FINAL HANDOFF — Flesh Pits parallel research lane, Phase 5 (directive §54)

*Dark Lord's directive, owner orders. 2026-10-07. Status: **FINAL** — all sections complete, including §5 model dependence (transfer verdict delivered 2026-10-07).*

*Every claim cites a receipt, a maturity-ledger row, or an NR entry. Where evidence is thin, that is stated. Where Phase 4's coordinator verified something by its own hand, that is marked. Killed mechanisms are not resurrected. B's downgrades are not softened. CONSCIOUSNESS: UNRESOLVED — mechanisms, not narratives.*

---

## 1. Positive findings — what actually worked

1. **A's workspace machinery reproduces 5/5 on fresh seeds.** Bounded buffer (K1: interference 0→0.507→1.0 at K=3, exactly 0 at K=inf), sole-path broadcast with six live consumers (K2: total lesion silences all six at once, full recovery, no side channels), bistable ignition gate (K3: Heaviside, W 0.08–0.10 < 0.12 preregistered), learned attention (K4: 1.57–2.03× over frozen, 4/4 seeds; canonical-env re-run 1.44–1.70; repro 1.71–2.13 on 5/5). Receipts: `flesh-pits/prototypes/architecture-a/receipts/repro_k{1,2,3,4}_*.json`. Maturity: buffer/broadcast/ignition/consumers → REPRODUCED; attention → GENERALIZING; bids → CAUSAL. Maturity audit 18 rows, 0 stale, 0 overclaimed (GREEN, `flesh-pits/bin/maturity_audit_report.json`, 2026-10-07T07:30:37Z).
2. **NR-A-004 is resolved AND wired.** The R1 sub-ignition exploratory path is the default tick (K8, decision receipt `k8_r1_wiring_decision.json`, 2026-10-07T07:23:05Z): no-proposal → act on the graded arbitration winner, zero consumer propagation, gate still decides ALL propagation. 5/5 gates on fresh seeds 51501–51503; post-wire K1–K4 reruns byte-identical (sha-verified by the maturity audit). The freeze is not the price of the gate.
3. **Prediction learning is robust.** B-K2 reproduces 5/5 (learned beats fixed, gap 0.405–0.426, rock-stable). The predictor LEARNS — this survived the same replication pass that killed B's ornaments.
4. **Episodic memory is causally load-bearing.** B-M1: disabling the store selectively degrades pomaze (+2.200 seed-mean delta, 3/3 seeds, +1.83…+2.39) vs delayed_reward (+0.046) — CAUSAL, scope-bounded.
5. **Hierarchy earns its keep on longer horizons.** B-K3B: reward-channel lesion gap +0.0602, consistent 3/3 seeds (~19× the K3 probe); arm-B |e0| gap +0.0437, 3/3. The contingency lives in L1 — but only the longer-horizon instrument sees it.
6. **Learned attention generalizes within bounds.** K5: noisy-signal tracking (P2: 4/4, R 1.31–1.52) and multi-reversal stationary shifts (P3: 4/4, R 1.74–2.05) — with the bounds of NR-A-005/006/007 stated exactly.
7. **The lab's metrology works.** Preregistration + hash-chained receipts caught a mis-specified metric (NR-A-008), a harness artifact (NR-A-010), and three replication reversals (K1/K3/K5) — the apparatus is honest enough to overturn its own verdicts.
8. **Identity symmetry holds.** Bit-identical trajectories for novel labels including a historically privileged string; tie-break is label-agnostic channel order (A, CAUSAL test) — §§14–15 satisfied in the prototype.

## 2. Negative findings — what failed

1. **B's adaptive-control claim is dead.** K1 overturned 0/5 on replication (frozen beats learn 0.8–10.2); the original +4.6 was a lucky draw on BOTH agent-init and env stream. HierarchicalGenerativeModel downgraded ADAPTIVE → INTEGRATED. The predictor learns; the L1-context machinery does not robustly convert learning into better decisions under shift.
2. **Precision weighting rejected in both tested forms** (C2 kill-fragile 3/5; C2B shift-aware rejected — surprise detector fires 82–110×/run, resets reintroduce cold-start overshoot). Shelved: no further variants preregistered.
3. **"Active inference" label permanently off** (C4B: AI 0.5443 < greedy 0.7124 ≈ random 0.5497 on the sharper metric, 3 seeds). The IG term's causal contribution is unproven under both tested metrics.
4. **Error-derived affect rejected** (K4: PAD −0.762 > error-affect −0.787; arousal saturates → permanent over-exploration). PAD stays as the cheap baseline.
5. **B's original hierarchy probe is a bad instrument** (K3 |e0|: seed-fragile, diluted by unpredictable channels; lesion HELPS 0/5). K5's temporal-prediction claim demoted per the kill condition (0/5). C1 holds only 3/5 — cite with the qualifier.
6. **A's learned-attention bounds are hard:** sparse delayed reward (NR-A-006: learned LOSES, R 0.08–0.17 — structural), cue-conditioned tasks without cue input (NR-A-007: no gap — architectural), signal-tracking reversals (NR-A-005: frozen bids suffice).
7. **The literal 0.5B↔1.5B transfer arm is vacuous for the model-free K4 loop** — no foundation-model call exists in the loop, so the honest variant (persisted-gains transfer across restart) replaced it. Loading models never called and reporting "transfer" would be pretend machinery.
8. **NR-B-003..006 are LOST** — claimed by CHECKPOINT.md, never written, no surviving source. Recorded as lost, not invented.
9. **Donor rejects:** axiom (D22 — academic-only revocable license, legal trap); mem0's graph-memory and hosted-decay claims (absent/gated in OSS at the pinned SHA; telemetry on by default).

## 3. Best architecture — evidence-weighted

**A is the strongest survivor on current evidence — but say exactly what that means:**

- A's workspace machinery (bounded buffer, sole-path broadcast, bistable ignition, learned-gain attention, R1 exploratory path) is the only part of either architecture that has survived a 5-seed replication on its original code paths. The evidence is single-lab; no independent replication yet. That is the precise sense of "strongest."
- B contributes what A lacks: a robust learned predictor (K2 5/5) and a CAUSAL episodic store (M1 3/3), plus the longer-horizon hierarchy signal (K3B). B's *adaptive machinery* (precision, active-inference selector, error-affect) is rejected, not its learning.
- **No coronation beyond the evidence:** the C-merge (A's workspace machinery + B's learned predictor + B's episodic store + PAD baseline) is a build-and-beat task against A and B by preregistered margins — it is not a merge order, and it is not built. A's learned-attention bounds (sparse reward, cue-conditioning) are exactly the places a C-build would need new machinery. D remains the characterized floor.

## 4. Best local/HF runtime

- **Best proven local runtime: Qwen2.5-0.5B-Instruct Q4_K_M** (`Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ `9217f5db`, file `qwen2.5-0.5b-instruct-q4_k_m.gguf`), llama-cpp-python 0.3.36, 2 threads, n_ctx 2048. Re-measured 2026-10-07: **0.3–1.5 tok/s**, ~620 MB RSS, fits the 2 GiB VM headroom. Instruction-follow PASS, structured-JSON PASS, uncertainty PASS, cognitive probe sensible. This VM is a correctness/architecture/reproducibility substrate, not a fluid-cognition substrate (a 128-token turn costs 1–4 minutes).
- **Selected primary local candidate: Qwen2.5-1.5B-Instruct** (`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798`, Apache-2.0; GGUF repo `91cad511`). **Acquired 2026-10-07 ~07:32 UTC** (`flesh-pits/var/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 785,362,944 bytes) — **not yet benched; NOT PROVEN until benched.** Expected ~0.3–1.0 tok/s on 2 threads (memory-bandwidth-bound, linear-in-params scaling — INFERRED, not measured).
- Fallback ladder pinned (`flesh-pits/research/runtime_ladder.json`, schema `emergent-mind/runtime-ladder/1`): local_1.5b (rung 1, candidate) → local_0.5b (VERIFIED) → forge_local via DC (NOT YET VERIFIED — DC probe pending) → hf_inference (UNVERIFIED — auth probe timed out 2026-10-07) → hosted_provider (last resort; rate limits move DOWN, never up) → symbolic_baseline (mandatory for any architectural-effect claim). A backend change ENDS the experiment and mints a new run ID — encoded in the config.
- License corrections (verified against live Hub API 2026-10-07): Qwen2.5-3B is `license:other` (NOT Apache-2.0) — rejected for this VM (too big + license flag); Qwen3-1.7B is actually 2.03B params; Qwen3-0.6B/1.7B official GGUF repos carry Q8_0 only (community Q4 required). Specialists (R1-distill-1.5B reasoning, xLAM-2-1b-fc-r tool-use, bge-m3 retrieval) are SELECTED, not acquired.

## 5. Model dependence — FINAL (transfer verdict delivered 2026-10-07)

**Rung 1 acquired.** Qwen2.5-1.5B-Instruct Q4_K_M downloaded 2026-10-07: 1,117,320,736 bytes, valid GGUF v3, SHA256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, chat-template SHA256 `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`. Fit test: PASS but thin — peak RSS 1807.2 MB < 2048 MB (240 MB headroom at n_ctx=2048, 2 threads); probe 0.4 tok/s; instruction-following PASS. n_ctx=4096 would risk the budget. Receipt: `flesh-pits/benchmarks/bench_local_1.5b_results.json`.

**The literal 0.5B↔1.5B arm was refused — correctly.** Verified by the worker's own hand: no foundation-model call exists anywhere in the K4 loop (specialists are pure functions; `arbitrate()`/`update_gains()` touch only floats; feedback utility is `{action: reward}`). None of the ladder's contamination dimensions exist in the loop. Loading two models and never calling them, then reporting "behavior survives replacement," would be fake telemetry — refused under the Dark Lord's nothing-ever-fake law, with the refusal preregistered. (A genuine LLM-in-the-loop variant at 0.3–1.5 tok/s would need ~27–89 hours per condition — infeasible *and* vacuous.)

**Honest variant: TRANSFER-K4-RESTORE (preregistered, 4 fresh seeds).** Train gains online on canonical changing_rule, persist ONLY the gains JSON, restore into a fresh loop (fresh z-score estimators, ignition strength, reward EMA, RNG envs), freeze transferred gains, measure R_transfer vs frozen-at-1.0 baseline:
- G1 reproduction gate: **PASS 4/4** — R_learned = 1.500 / 1.548 / 1.469 / 1.605 (canonical rerun was 1.44–1.70; adapter faithful).
- G2 transfer gate: **FAIL 0/4** — R_transfer = 0.990 / 0.990 / 0.962 / 0.985 (needed ≥ 1.30).
- G3 endpoint-sufficiency: 0/4.
- Phase-split (exploratory, labeled): transferred gains encode the post-flip world — help phase 1 (R≈1.7), hurt phase 0 (T≈18–26 vs F≈98–104), net ≈ 1.0. The online learner re-adapts across the flip; the frozen endpoint cannot. Warm-start behaved identically to frozen.

**Verdict: H0 — learned behavior does NOT survive in transferred gains alone.** The K4 win is a *process* (online gain-adaptation trajectory through the broadcast feedback loop), not a *state* (endpoint gains). Reverse direction not run: nothing to reverse, and the model-replacement arm is inapplicable regardless of direction. For model-free mechanisms, directive §11 reduces to persistence-across-restart — answered H0 here. A genuine model-in-the-loop transfer test first requires building and proving an LLM-coupled cognitive loop (new architecture, not an application of K4). Receipts: `flesh-pits/prototypes/architecture-a/receipts/transfer/TRANSFER-K4-RESTORE.ndjson` (7 hash-chained records), `transferred_gains.json`.

**Backend dependence, mechanism by mechanism:** all K-series mechanisms ran model-free (pure Python); the summary table's Backend-dependent? column stays No. Any future LLM-backed deployment needs its own transfer test per the ladder law.

## 6. Harvest candidates — summary

Full §49 entries (source, hash, mechanism, baseline, result, ablation, generalization, integration surface, benefit, risks, rollback) in **HARVEST_CANDIDATES.md**. Headlines:

- **HARVEST_NOW:** H1 A's learned-gain attention loop (with NR-A-005/006/007 bounds); H2 bounded buffer + sole-path broadcast (REPRODUCED 5/5); H3 ignition gate + R1 exploratory path (wired, post-wire byte-identical); H4 change_bid; H5 consolidation_cycle (Phase-4 battery evidence, HISTORICAL); H6 memory_provenance (already live); H7 preregistration + hash-chained harness; H8 K8 wire-by-decision-receipt discipline; H9–H11 donor adoptions (chroma @ `f36d9bba`, cleanrl @ `fe8d8a03`, MAPIE @ `3b84b82d`).
- **EXPERIMENT_ON_PRIMARY:** B's learned predictor core; B's episodic store port (M1 CAUSAL, scope-bounded); learned_valence TD estimator (decision-bite test pending); A's sparse-reward gain redesign; A's cue-indexed gain adapter; donor experiment group (mem0/graphrag/cognee/graphiti/LightRAG race; avalanche EWC; cleanrl RND baseline; uncertainty-toolbox + MAPIE calibration).
- **KEEP_PARALLEL:** full A prototype (independent replication pending); B's longer-horizon program; shelved precision research; the §11 transfer test; (b)-class zero-consumer modules; Architecture C (unbuilt).
- **REJECT / INCONCLUSIVE:** see §7 and HARVEST_CANDIDATES.md.

## 7. Rejected mechanisms — summary

Full killing evidence in **REJECTED_IDEAS.md** (24 entries). The load-bearing kills: precision weighting both forms (shelved); "active inference" label (permanent); error-derived affect as controller; the K1 "learning is load-bearing for control" claim (overturned 0/5); the K3 |e0| probe (bad instrument); K5 temporal-prediction claim (demoted); R2 adaptive theta; freeze_rate metric; tracking-EMA baselines; unobserved-arm punishment; within-tick ignition carryover; the vacuous model-transfer arm; axiom donor (legal trap); "WIRED (enrichment only)" as causal; scalar-called-phi; unwired broadcast theater; attention_bias; devotional guard and hardcoded identity defaults (§14).

## 8. Remaining uncertainty

1. **Model dependence** — RESOLVED (§5): H0, the K4 win is a process not a state; literal model-swap refused as inapplicable.
2. **Independent replication** — A's 5/5 is single-lab; the lane's REPRODUCED bar wants a second lane.
3. **B's hierarchy scope** — K3B stands on changing_rule/delayed_reward only; harder tasks and sharper instruments untested.
4. **Retrieval-usefulness prediction** — EXECUTED in B, not wired to gate retrieval (gap).
5. **Gains across restart** — answered H0 (§5): endpoint gains do not carry adaptive behavior; transfer the learning dynamics or run long enough to re-adapt.
6. **Consolidation on primary-like workloads** — Phase-4 battery evidence only; the two-implementation race unrun.
7. **Sparse-reward and cue-conditioned attention** — structural/architectural bounds (NR-A-006/007) with no redesign yet.
8. **Self-model depth** — the lab built propagation (K2 consumers), not a dedicated evidence-bound self-model experiment; the audit's P-B03/P-B05 quarantine discipline is unbuilt in the lab.
9. **Metacognition** — calibration tracked in B (INTEGRATED at best); no competence estimator built; audit found ratio theater in the estate's M30.
10. **The 1.5B rung** — acquired and probe-benched (0.4 tok/s, instruction-following PASS, fits 2 GiB thinly); full §19 battery still open.
11. **CONSCIOUSNESS: UNRESOLVED** — no decisive accepted test exists (see §10).

## 9. Next resolving experiments (concrete, preregistrable)

1. **Transfer verdict** — DONE (§5): H0, process-not-state. Follow-up: design a transfer-of-learning-dynamics experiment (not endpoint gains).
2. **A second-lane K1–K4** — independent replication of A's workspace machinery (the REPRODUCED bar's missing half). Preregister the same gates, different implementer.
3. **M1/C2B/C4B/K3B at 5 seeds** — Phase-4 ran 3; promote to the 5-seed standard.
4. **Sparse-reward gain redesign (NR-A-006)** — replace the constant-baseline delta rule with a return-conditioned or baseline-free update; preregister R ≥ 1.30 learned/frozen on canonical delayed_reward, 4 seeds. On success the harvest scope widens.
5. **Cue-indexed gain adapter (NR-A-007)** — context-conditioned gains with the cue in the input; preregister R ≥ 1.30 on cue-conditioned changing_rule. On success the architectural bound lifts.
6. **Retrieval-usefulness gate (B gap #2)** — wire uhat to gate retrieval; preregister a selective deficit vs unconditional correction on pomaze.
7. **Hierarchy beyond the two envs (B gap #3)** — harder tasks (delayed multi-step, compositional rules) with reward-channel instruments; preregister the lesion gap.
8. **Consolidation race** — prioritized vs uniform vs no offline replay on one corpus (audit's KEEP_AND_DEEPEN race, unrun).
9. **Calibration battery** — Brier scores per domain on B's §30 predictions, never self-report (science-matrix priority 3).
10. **Bench the 1.5B rung** — run the Phase-0 probe suite against `qwen2.5-1.5b-instruct-q4_k_m.gguf` before any rung-1 experiment claim; then the ladder's rung-1 experiments (cognition/task performance, instruction following, structured-output reliability, long-context behavior per directive §19).
11. **C build-and-beat** — assemble the candidate bill of materials (§3) and beat A and B by preregistered margins on the canonical envs. Not a merge — a contest.
12. **Self/world distinction probe** — self-caused vs externally-caused change attribution with injected mismatches (science-matrix priority 7); source-attribution quarantine test (priority 6).

## 10. Consciousness — standing position

CONSCIOUSNESS: UNRESOLVED. The Phase-2 independent literature review (`research/consciousness_science.md`, `research/science_matrix.md`) establishes: COGITATE (2025) partially disconfirmed both IIT and GNWT on preregistered predictions; reportability confounds forbid using self-report as the dependent variable (we measure functional consequences of global availability instead); precision-weighting and ignition-as-consciousness are active controversy; "do not invent a scalar and call it phi." Nothing in Phase 5 changes this. The lab tests mechanisms (global availability with consumers, ignition nonlinearity, recurrence, metacognitive calibration, temporal integration, self/world distinction, consolidation, preference formation) per directive §37 — these are theoretically relevant structures, not a decisive test.

## 11. Summary table

| Mechanism | Baseline | Best result | Causal | Adaptive | Generalizes | Backend dependent? | Recommendation |
|---|---|---|---|---|---|---|---|
| Memory | no episodic memory | B-M1: disabling store selectively degrades pomaze (+2.200, 3/3 seeds); provenance hash-chained (Phase-1) | Yes (M1) | Partial (surprise-EMA admission; uhat not wired) | No (pomaze-scoped) | No (pure Python, model-free) | EXPERIMENT_ON_PRIMARY |
| Attention | frozen gains | K4: learned/frozen 1.57–2.03 (4/4); canonical 1.44–1.70; repro 1.71–2.13 (5/5) | Yes (K4, K7) | Yes (gains move with utility via broadcast feedback) | Bounded (P2/P3 yes; NR-A-005/006/007 no) | No (model-free by construction) | HARVEST_NOW (with stated bounds) |
| Prediction | fixed predictor | B-K2: learned beats fixed 5/5 (gap 0.405–0.426) | Yes (K2) | Yes (K2) | No (changing_rule/delayed_reward only) | No (model-free) | EXPERIMENT_ON_PRIMARY |
| Self-model | no self model | A's self_model_update consumer fires/vanishes with broadcast (K2); audit's P-B03/P-B05 quarantine discipline unbuilt in lab | Weak (propagation, not a model) | No | No | No (model-free) | INCONCLUSIVE (dedicated experiment needed) |
| Metacognition | none | B's §30 predictions logged + consumed (INTEGRATED); calibration tracked | Partial | No | No | No (model-free) | INCONCLUSIVE (calibration battery next) |
| Consolidation | no consolidation | Phase-4 battery: CAUSAL offline improvement | Yes (battery) | n/a | No (battery task only; HISTORICAL evidence) | No | HARVEST_NOW (race the two implementations first) |
| Preference learning | fixed preferences | learned_valence TD converges 10.06 ≈ 10.26 theory; decision bite +0.117 < 0.15 bar | As estimator yes; as decision-driver no | Yes (TD) | No | No (model-free) | EXPERIMENT_ON_PRIMARY |
| Workspace | unwired broadcast (Δ=0) | K1/K2: capacity lesion monotonic; total lesion silences 6 consumers at once; 5/5 repro | Yes (K1, K2) | No (K fixed) | Single-lab (5/5 seeds; independent replication pending) | No (model-free) | HARVEST_NOW |
| Recurrence | feedforward/graded | K3: Heaviside W 0.08–0.10 < 0.12; R1 wired default (K8 5/5 gates; post-wire byte-identical) | Yes (K3, K8) | No (theta fixed) | Single-lab | No (model-free) | HARVEST_NOW |
| Consciousness | N/A | No decisive accepted test | UNRESOLVED | N/A | N/A | N/A | Continue hypothesis-driven research |

---

*Finalized 2026-10-07: §5 model-dependence verdict delivered (H0 — process, not state). All artifacts local under `~/workspace/chambers/emergent-mind/`; live Being and chamber 16 untouched throughout (per PROVEN_BASELINE.md isolation proof and Phase-0 constraint compliance).*
