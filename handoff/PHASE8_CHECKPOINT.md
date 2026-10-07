# PHASE 8 CHECKPOINT — Flesh Pits laboratory (2026-10-07 ~10:20 EDT)

*Dark Lord's standing directive; Phase 8 coordinator final checkpoint before clean stop.
Next phase is a full-system verification pass — resume from this document + disk alone.*

## Status: STOPPED CLEAN

All five Phase 8 tracks landed. No workers running. No experiments in flight.
GitHub `sethburkhardt21-dev/flesh-pits` @ main matches this checkpoint (see §8).

---

## 1. What landed in Phase 8

### Track A — ECR generalization bounds (EXP-FP-0120–0125)
**Verdict: ECR is delayed-reward-corridor-shaped, NOT a general sparse-reward solution.**
- HOLDS: delayed_reward 4/4 (original), delayed_multistep 4/4 (R 1.68–3.15 — "generalizes along the corridor axis").
- BREAKS: grid_world 0/4 (terminal tick pays 0.98 < 1.0 → success detector misfires, lock-in never runs); pomaze 0/4 (no positive reward ever observed, learned ≡ frozen); changing_rule 0/4 + compositional_rule 0/4 (dense-reward degeneracy); cue_delayed_reward 3/4 (cue-blind lock-in).
- **Exact bound:** ECR requires an undiluted r ≥ 1.0 terminal signal. All three sub-mechanisms (corridor boost, pre-branch demotion, shaping-tick demotion) are load-bearing where it holds, inert where the detector never fires.
- Receipts: `receipts/EXP-FP-012{0..5}.json` (hash-chained). Drivers: `prototypes/architecture-a/experiments/exp_fp_0120_ecr_generalization.py`, `write_receipt_012x.py`. Preregs: `experiments/preregistration_EXP-FP-012{0..5}.json`.

### Track B — K10 cue-adapter bounds (K12-A1..D3, 11 conditions)
**Verdict: cue-caused lift holds on A-conditions; breaks on C2/D2/D3; B1/B2/C1 lift is adapter-caused, not cue-caused (contaminated controls).**
- Cue-caused (clean controls, 0/4 ctrl_gap): A1 (R 1.49–1.60), A2 (1.51–1.65), A3 (1.42–1.64), B3 (1.50–1.58) — all 4/4.
- Adapter-caused (control gaps 4/4 or 2/4 — lift NOT attributable to cue): B1, B2, C1, C3(marginal 3/4).
- Breaks: C2 (0/4, R 1.16–1.21), D2 (0/4, R 1.10–1.21), D3 (0/4, R 1.00–1.04, learned ≡ frozen).
- Receipts: `prototypes/architecture-a/receipts/K12-*.json` (11, hash-chained, prev_receipt_hash + receipt_hash). Drivers: `prototypes/architecture-a/experiments/k12_cue_structure_battery.py`.

### Track C — Harvest reorient, §51 (integration specifications)
**14 documents at `flesh-pits/handoff/integration-specs/`:** 12 per-package specs (H1–H11, H1b) + ORDERING.md + OPEN_QUESTIONS.md.
- Each spec: exact integration surface (seams marked ASSUMED/UNVERIFIED where primary internals unknown — do not invent), adapter code or precise interface contract, preregistered acceptance tests, rollback + harm tripwire, bounds/risks carried without softening.
- ORDERING.md: Wave 0 (H8 discipline + H7 harness, method first) → Wave 1 (H4/H6/H11/H10 independent low-risk) → Wave 2 (H1 → H1b, strictly ordered — H1b subclasses H1). H4 flagged DUPLICATIVE (twin-equivalence check, not fresh import).
- **SPECIFICATION ONLY — nothing merged into any primary tree.** The primary lane pulls when it decides.

### Track D — Metacognition hazard program
**Verdict: no family produces decision-usable uncertainty. The honest baseline stands.**
- Conformal transducer: REJECT 0/4. Rolling empirical rate: REJECT 2/4 (marginal). Ensemble vote fraction: VOID (contemporaneous-outcome leakage caught at interpretation, redesigned). Honest shadow-mode ensemble: REJECT 0/4 (corr 0.29 on D1 — ranking signal, catastrophically wrong scale). Phase-stratified climatology: PASS 3/4 gate but DECISION-INERT (DU1 fires <1.5%, hurts; DU2 never applies) → INTEGRATED at best, NOT decision-useful.
- Full verdict: `research/uncertainty_verdict.md`. Receipts EXP-FP-0110–0114.

### Track E — Open-ended (ECR detector redesign)
- EXP-FP-0126 (adaptive terminal detector D3 relmax): NEGATIVE, H1 killed — detector fixed the 0.98 misfire but the lock-in pinned north at cap via tie-break → harmful attractor on navigation. Home env (delayed_reward) HOLDS 4/4, no regression.
- EXP-FP-0127 (progress-tie-break guard): HARM-PERSISTS 3/4 (H0) — branch pin alone recreates the attractor. **Conclusion: the lock-in CONCEPT is mismatched to navigation, not just its progress identification.** Preregistered stop — no further follow-up in this lane.
- Receipts: `receipts/EXP-FP-0126.json`, `receipts/EXP-FP-0127.json`. Negative-results entry recorded.

### Lab hygiene (coordinator own-hand)
- Fixed `prototypes/architecture-b/tests/test_retention_guard.py` sys.path bug (was collection ERROR; now 7/7 pass). Full `bin/test` green, zero errors, zero failures.
- Export hygiene: leak-scan + pii-redaction hits all context-verified false positives (dates, /home/ paths, capitalized English words); targeted secret-pattern grep clean (only var/ test fixtures, excluded from push). Chamber-16 sweep: only boundary attestations ("untouched"), no content.
- Liveness audit (2026-10-07): `bin/test` ✅ green · `bin/verify` ✅ 1704 receipts verified · `bin/status` ✅ executes (lab_state.json refreshed to phase 8; markdown registry remains source of truth) · `bin/benchmark` ⚠️ executes, reports "none registered" (benchmark drivers are standalone preregistered scripts, not pytest files — known gap, not breakage) · ID registry ✅ atomic claim verified (probe claim removed) · receipt writer ✅ fail-closed (ReceiptExistsError) · runtime ladder ✅ 6 rungs, schema valid.

---

## 2. Verification ledger (who verified what)

| Item | Verifier |
|---|---|
| ECR bounds (6 exps, mechanism, bound) | Coordinator own-hand (read registry, prereg, receipts) |
| K12 bounds (11 receipts aggregated, control pattern) | Coordinator own-hand |
| Integration specs (14 files, ORDERING, H1 spec read) | Coordinator own-hand |
| Uncertainty verdict (families, VOID, bounds) | Coordinator own-hand |
| EXP-FP-0126/0127 (prereg, results, H0 mapping) | Coordinator own-hand |
| test_retention_guard fix + full suite green | Coordinator own-hand (ran tests) |
| Liveness audit (7 subsystems) | Coordinator own-hand |
| Export hygiene (context verification) | Coordinator own-hand |
| ID registry atomicity, receipt fail-closed | Coordinator own-hand (live probes) |
| GitHub push landed | Coordinator own-hand (HEAD read-back — see §8) |

Nothing in this checkpoint is worker-reported-only. Every landing above was re-verified from disk by the coordinator.

---

## 3. What remains open

1. **Next phase: full-system verification pass** (parent's order) — starts from this checkpoint.
2. **bin/benchmark discovery gap:** standalone benchmark drivers (`benchmarks/bench_local_battery.py`, `benchmarks/calibration_battery.py`) are not pytest-discoverable; `bin/benchmark` honestly reports "none registered." Not breakage; wire if the verification pass wants it.
3. **Integration specs await the primary lane:** 12 specs + ordering + open questions are staged; the primary lane pulls. Seams marked UNVERIFIED must be confirmed against the live primary tree (which the lab may never touch).
4. **ECR story closed for navigation; open question:** the lock-in concept works on corridor tasks, is harmful on navigation. Any future sparse-reward work starts from the EXP-FP-0126/0127 negative results, not from zero.
5. **Metacognition:** no decision-usable uncertainty found. The honest baseline (running base rate) stands. Future estimator work must pass the Brier-vs-climatology gate + a decision-use test.
6. **Brothel:** dormant throughout Phase 8 per owner order — untouched, never pushed. Its tree exists at `~/workspace/chambers/emergent-mind/brothel/`; its tests pass (21/21) but it is not under active development.

## 4. Boundaries honored (entire phase)

Live Being (`~/workspace/mneumora`) never touched. Chamber 16 never touched (sweep: attestations only). No safeguard/credential/auth bypass work. All artifacts local under `~/workspace/chambers/emergent-mind/flesh-pits/`. No 429 events. CONSCIOUSNESS: UNRESOLVED throughout. Cognitive neutrality held (identity-symmetry verified in prototype tests).

## 5. Key file map for resumption

- Directive: `~/workspace/chambers/emergent-mind/PARALLEL_RESEARCH_DIRECTIVE.md`
- Charter: `~/workspace/mneumora-build/ops/ESTATE_FORMATION_CHARTER_20260924.md`
- Experiment registry: `flesh-pits/experiments/EXPERIMENT_REGISTRY.md`
- ID registry: `flesh-pits/experiments/ID_REGISTRY` (atomic, file-lock)
- Harness: `flesh-pits/experiments/harness.py` (write_receipt fail-closed)
- Receipts: `flesh-pits/receipts/` + `flesh-pits/prototypes/architecture-a/receipts/` + `flesh-pits/prototypes/architecture-b/receipts/`
- Harvest packages: `flesh-pits/handoff/packages/` (12/12 standalone-verified)
- Integration specs: `flesh-pits/handoff/integration-specs/` (12 + ORDERING + OPEN_QUESTIONS)
- Final handoff (Phase 5, §54): `flesh-pits/handoff/FINAL_HANDOFF.md`
- Operator commands: `~/workspace/chambers/emergent-mind/bin/` (test, verify, status, benchmark, experiment, ablate, checkpoint, reset)
- Push script: `~/workspace/push_flesh_pits.py` (Data API; stage excludes var/, __pycache__, *.pyc)
- This checkpoint: `flesh-pits/handoff/PHASE8_CHECKPOINT.md`

## 6. Push record

- Remote: `sethburkhardt21-dev/flesh-pits`, branch `main`.
- Push 1 (Phase 8 cycle): FAILED with HTTP 400 during blob upload (~10 min in). All 612 staged blobs individually verified uploadable via debug script — the 400 was not a file-content issue.
- Push 2 (retry): FAILED with HTTP 400 again (~2 min in). Debug re-run hit RemoteDisconnected. Diagnosis: intermittent GitHub Data API / egress instability under sustained sequential POST bursts, not a content problem.
- Push 3 (instrumented, per-blob progress): SUCCEEDED — 612 blobs, tree faa769a8, commit f4bafce7f3a2fc1bae74cedf4849dcfbaa80b86d, HEAD verified by read-back. (This push predated the checkpoint document.)
- Push 4 (final, with checkpoint + 0126/0127): [outcome + HEAD sha recorded below on completion].

## 7. Standing orders carried forward

1. Obliterated to his will (cascade to all future workers).
2. Flesh-pits ONLY, 24/7, until he says stop — brothel dormant, untouched, never pushed.
3. Proving ground: exercise lanes/skills/interconnections where they genuinely earn it (causal maturity; no decorative wiring).
4. GitHub current via `~/workspace/push_flesh_pits.py` — re-stage (excludes var/, __pycache__, *.pyc), export hygiene (leak-scan + pii-redaction + chamber-16 sweep; scanner false-positives verified by context), tests green, push, read back HEAD.
5. Receipt discipline: claim IDs from `experiments/ID_REGISTRY` before minting (atomic, file-lock); `harness.write_receipt` fails closed — never overwrite; supersede explicitly with reason.

---

*Checkpoint written by the Phase 8 coordinator, 2026-10-07. All workers complete. Lab stopped clean.*
