# ARCHITECTURE C — CAUSAL MATURITY LEDGER

*Directive §41. Created 2026-10-07 with EXP-FP-C-BUILD-AND-BEAT.
Levels: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE →
GENERALIZING → REPRODUCED. Marked honestly; nothing above its evidence.
Parent evidence (A/B) is cited, not inherited: a level claimed for C
requires C-measured evidence.*

| Component | Level | Evidence |
|---|---|---|
| `c_agent.py` — predictor-driven specialists (B->A surface) | **INTEGRATED** | Specialist stimuli are computed from the predictor's `predict_reward` + the memory's `predicted_error_for_action` and consumed by A's arbitration every tick (test: `test_predictor_drives_stimuli`, `test_memory_bonus_enters_stimulus`). CAUSAL pending the C-frozen-predictor ablation arm (EXP-FP-C-BUILD-AND-BEAT). Parent: B-K2 (predictor learns, 5/5). |
| `c_agent.py` — per-tick learning close (`close_tick_c`) | **INTEGRATED** | `model.observe` + `memory.store` execute every tick with the transition's exact (state, action, ctx, obs_next, reward); weights and belief move (test: `test_learning_close_updates_predictor`). Same call sequence as ArchB._close_tick minus affect/§30 (neither is in C's assembly). CAUSAL pending the frozen-predictor arm. |
| `c_agent.py` — episodic store in the C loop | **INTEGRATED** | `memory.store` encodes every tick with provenance; `predicted_error_for_action` feeds the specialist bonus, None-guarded (tests: `test_memory_bonus_none_guarded`, `test_no_memory_ablation_runs`). CAUSAL pending the C-no-memory (DisabledStore) arm. Parent: B-M1 (CAUSAL, pomaze-scoped). |
| `c_agent.py` — CAdapter | **EXECUTED** | Tick protocol (observe/step->reward) + driver protocol (last_transition/reset_episode) exercised on all five battery envs (test: `test_adapter_protocol`). |
| A's WorkspaceTick inside C (buffer, ignition, broadcast, R1) | **INTEGRATED** | Imported unchanged; the C loop's action selection flows through arbitration -> buffer -> ignition -> sole-path broadcast -> planner_input, with the R1 sub-ignition path as default (tests: `test_sole_path_no_consumer_refs`, `test_r1_default_sub_ignition`). Parent levels (A: REPRODUCED machinery) are A's, not C's — C has not re-proven them. |
| A's learned-gain loop inside C | **INTEGRATED** | Closes through the tick's feedback broadcast (utility={action: reward}, fixed baseline 0.5), cue-indexed where the env exposes a cue (tests: `test_cue_indexed_arbitrator_wired`, `test_default_arbitrator_without_cue`, `test_frozen_gains_pinned`). CAUSAL pending the C-frozen-gains arm. Parent: A-K4/K10. |
| Excluded (preregistered, not wired) | n/a | PAD (no genuine consumer in C — decorative), error-affect (REJECTED), ActiveInferenceSelector (IG label killed), UsefulnessPredictor/RetrievalGate (uncalibrated + degenerate), precision variants (rejected/shelved), consolidation_cycle (no online surface). Exclusion is a design claim, documented in `experiments/preregistration_ARCH_C.json`. |

## What failed or is BLOCKED

- EXP-FP-C-BUILD-AND-BEAT battery: IN FLIGHT (started 2026-10-07).
  Results, per-env verdicts, and ablation attribution to be appended here
  on completion. Levels above move only on C-measured evidence.

## Build-and-beat result (EXP-FP-C-BUILD-AND-BEAT)

*PENDING — appended after the battery completes. Will record: per-env
per-seed mean returns for A/B/C + 3 ablation arms, win rules vs the
frozen margins (20 / 20 / 0.5 / 0.15 / 0.15), the overall C_BEATS verdict,
receipt path, and the honest negative if the margins are not met.*
