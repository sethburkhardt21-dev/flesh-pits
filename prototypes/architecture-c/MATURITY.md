# ARCHITECTURE C — CAUSAL MATURITY LEDGER

*Directive §41. Created 2026-10-07 with EXP-FP-C-BUILD-AND-BEAT.
Levels: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE →
GENERALIZING → REPRODUCED. Marked honestly; nothing above its evidence.
Parent evidence (A/B) is cited, not inherited: a level claimed for C
requires C-measured evidence.*

| Component | Level | Evidence |
|---|---|---|
| `c_agent.py` — predictor-driven specialists (B->A surface) | **INTEGRATED** | Specialist stimuli are computed from the predictor's `predict_reward` + the memory's `predicted_error_for_action` and consumed by A's arbitration every tick (test: `test_predictor_drives_stimuli`, `test_memory_bonus_enters_stimulus`). NOT CAUSAL in C: the C-frozen-predictor ablation (changing_rule, 4 seeds) scores mean 30.65 >= C's 30.04 — the predictor contributes nothing measurable and slightly perturbs the gain loop. Parent B-K2 (predictor learns, 5/5) is B's evidence, not C's. |
| `c_agent.py` — per-tick learning close (`close_tick_c`) | **INTEGRATED** | `model.observe` + `memory.store` execute every tick with the transition's exact (state, action, ctx, obs_next, reward); weights and belief move (test: `test_learning_close_updates_predictor`). Same call sequence as ArchB._close_tick minus affect/§30 (neither is in C's assembly). The close is executed and consumed; its behavioral contribution in C is unproven (see specialists row). |
| `c_agent.py` — episodic store in the C loop | **INTEGRATED** | `memory.store` encodes every tick with provenance; `predicted_error_for_action` feeds the specialist bonus, None-guarded (tests: `test_memory_bonus_none_guarded`, `test_no_memory_ablation_runs`). NOT CAUSAL in C: C-no-memory (DisabledStore) mean 29.85 ~= C 30.04 on changing_rule, ~= -5.87 vs -5.80 on pomaze — no measured contribution. Parent B-M1 (CAUSAL, pomaze-scoped) is B's evidence, not C's. |
| `c_agent.py` — CAdapter | **EXECUTED** | Tick protocol (observe/step->reward) + driver protocol (last_transition/reset_episode) exercised on all five battery envs (test: `test_adapter_protocol`). |
| A's WorkspaceTick inside C (buffer, ignition, broadcast, R1) | **INTEGRATED** | Imported unchanged; the C loop's action selection flows through arbitration -> buffer -> ignition -> sole-path broadcast -> planner_input, with the R1 sub-ignition path as default (tests: `test_sole_path_no_consumer_refs`, `test_r1_default_sub_ignition`). Parent levels (A: REPRODUCED machinery) are A's, not C's — C has not re-proven them. |
| A's learned-gain loop inside C | **CAUSAL** (changing_rule-scoped) | Closes through the tick's feedback broadcast (utility={action: reward}, fixed baseline 0.5), cue-indexed where the env exposes a cue (tests: `test_cue_indexed_arbitrator_wired`, `test_default_arbitrator_without_cue`, `test_frozen_gains_pinned`). CAUSAL in C: C-frozen-gains collapses changing_rule mean 30.04 -> 22.00 (4/4 seeds worse), the only ablation with a measured behavioral effect. Scope: changing_rule only — on pomaze all arms sit at the floor (~-5.9), undifferentiated. Parent: A-K4/K10. |
| Excluded (preregistered, not wired) | n/a | PAD (no genuine consumer in C — decorative), error-affect (REJECTED), ActiveInferenceSelector (IG label killed), UsefulnessPredictor/RetrievalGate (uncalibrated + degenerate), precision variants (rejected/shelved), consolidation_cycle (no online surface). Exclusion is a design claim, documented in `experiments/preregistration_ARCH_C.json`. |

## What failed or is BLOCKED

- EXP-FP-C-BUILD-AND-BEAT battery: IN FLIGHT (started 2026-10-07).
  Results, per-env verdicts, and ablation attribution to be appended here
  on completion. Levels above move only on C-measured evidence.

## Build-and-beat result (EXP-FP-C-BUILD-AND-BEAT) — COMPLETE 2026-10-07

**Verdict: FAIL — C does not beat A or B by the preregistered margins.
A clean negative; the hybrid's added complexity is not justified.**

Per-env (4 fresh seeds 80101–80104; margins 20 / 20 / 0.5 / 0.15 / 0.15):

| env | A mean | B mean | C mean | C_vs_A | C_vs_B |
|---|---|---|---|---|---|
| changing_rule | 31.10 | 22.40 | 30.04 | d=-1.06, 0/4 → FAIL | d=+7.65, 4/4 but < 20 → FAIL |
| compositional_rule | 35.46 | 27.00 | 35.08 | d=-0.38, 0/4 → FAIL | d=+8.08, 4/4 but < 20 → FAIL |
| pomaze | -3.95 | -3.62 | -5.80 | d=-1.85 → FAIL | d=-2.18 → FAIL |
| delayed_reward | 0.0628 | 0.0773 | 0.1925 | d=+0.130, 3/4 but < 0.15 → FAIL | d=+0.115, 3/4 but < 0.15 → FAIL |
| cue_delayed_reward | 0.0668 | 0.0548 | 0.0325 | d=-0.034 → FAIL | d=-0.022 → FAIL |

Overall: C_BEATS_A = False (0/5), C_BEATS_B = False (0/5), C_BEATS = False.

Ablation attribution (changing_rule + pomaze, descriptive): the gain loop
is the only load-bearing surface in C (C-frozen-gains collapses to 22.00);
the predictor (C-frozen-predictor 30.65 ≥ C 30.04) and the memory
(C-no-memory ≈ C) contribute nothing measurable. On pomaze all arms sit at
≈ -5.9 — the failure is architectural: flat predictor outputs habituate
the bids, gains decay to floor under sparse negative reward (the inherited
NR-A-006 structural decay), arbitration degenerates to tie-break.

Receipts: `prototypes/architecture-c/receipts/EXP-FP-C-BUILD-AND-BEAT.ndjson`
(97 records, hash chain verified) + `.summary.json`. Preregistration:
`experiments/preregistration_ARCH_C.json` (frozen before implementation).
Gates: G0a tripwire CLEAN, G0b chain verified, G0c determinism MATCH
(1e-9), G0d paired seeds/driver, G0e A/B unmodified by this lane.

*The "PENDING" note above is now closed.*
