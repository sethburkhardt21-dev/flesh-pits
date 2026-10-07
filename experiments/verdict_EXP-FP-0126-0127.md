# Verdict page — EXP-FP-0126 / EXP-FP-0127: adaptive terminal detector for ECR

*Track E, 2026-10-07. One deep hypothesis + one preregistered follow-up. Both died cleanly.*

## The hypothesis

EXP-FP-0120 proved that ECR's hardcoded `r >= 1.0` terminal detector silently misfires on grid_world: the goal tick pays −0.02 + 1.00 = **0.98 < 1.0**, so terminal success is absorbed as a shaping tick and the episodic lock-in (branch + progress cap-set) never executes — learned is bit-identical to frozen, 0/4.

H1 (EXP-FP-0126): a scale-relative detector — terminal iff `r > 0 AND r ≥ max(0.1, 0.5 × r_big)`, where `r_big` is the running max of positive per-tick rewards — restores the lock-in on grid_world **and** preserves the home env (delayed_reward: shaping 0.02 < 0.1 stays shaping; terminal 1.0 fires). Two-sided gate: grid_world 4/4 goal-episodes AND delayed_reward 4/4 R ≥ 1.30, fresh seeds, canonical files untouched.

**Why this hypothesis:** Phase 8's cleanest single-line mechanism break; falsifiable; two-sided gate; the alternative directions had already landed via sibling lanes (EXP-SW-01/02, EXP-FP-0060, EXP-FP-0040). Cheapest experiment in the menu per unit of load-bearing surface.

## What happened

**The detector fix works mechanically.** Gain-history inspection on seed 88101 shows the single 0.98 goal tick flagged `terminal=True` — the first time any ECR variant has ever fired its lock-in on grid_world.

**The lock-in it enables is actively harmful.** The firing tick cap-set `branch_credit=north` (first branch winner) + `progress_credit=north` (canonical tie-break — zero shaping receipts that episode). Final gains `{north: 2.0, rest 1.0}`: the agent became a north-attractor with no unlearning path (non-goal ticks on grid_world are negative/zero and change no gains), bumping into the north wall for the rest of the run. Result: **BREAKS 0/4** — learned goals 1/1/1/0 vs frozen 2/0/3/1; learned totals more negative than frozen on 3/4 seeds. The home env held 4/4 (R 4.28/2.06/7.82/12.33) — the detector is safe where designed.

**Ablations:** D0 (hardcoded) reproduces the 0120 failure bit-identically on fresh seeds; D1 (floor-only) and D2 (k×baseline) are gain-identical to D3 on both envs — the floor does all the work, relativity adds nothing on these reward streams. Detector variants are not the load-bearing question.

## The follow-up (EXP-FP-0127)

The failure's sharpest question: is the arbitrary progress pin (tie-break) *the* harmful half? Preregistered guard: when an episode contains zero shaping receipts, lock in the branch choice only (skip the progress cap-set). No-harm gate on 4 fresh seeds.

**HARM-PERSISTS (3/4).** The guard works as designed (no arbitrary channel capped, `progress_credit=None` recorded), but on seed 88302 the branch pin alone (`north=2.0`, the first-branch winner) recreated the attractor: goals 1 vs frozen 2. The tie-break is **not** the sole harmful half — pinning *any* single movement channel at the gain cap builds an attractor, because every movement tick counts as a "branch" under the given prior and no unlearning path exists. Bit-identity on delayed_reward (3.080000/1.980000 exact) proves the guard is inert where shaping exists.

## The verdict

**ECR's episodic lock-in is a corridor-task mechanism. On navigation tasks it is not merely inert — it is actively harmful.**

The refined bound, stated exactly: the lock-in holds iff (1) terminal success is identifiable from the reward stream (an adaptive detector fixes the 0120 line — necessary but not sufficient); **and** (2) the locked channels are genuine decisions — a single branch choice per episode plus a shaping-identified progress channel; **and** (3) the environment provides an unlearning/correction path or the lock is correct-by-construction, because a wrong lock perseverates (0124's cue-lock, 0126/0127's navigation attractor). Where every tick is a "branch" and positives carry no progress information, the lock-in builds a one-direction attractor and underperforms frozen gains.

Any harvest candidate must carry all three boundary conditions. A "general ECR" would additionally require cue-conditioned gain vectors (0124), an unlearning path for wrong locks (0124/0127), and a progress-channel ID that degrades gracefully without shaping (0126) — each a new experiment, not a tuning.

## Maturity

- Adaptive detector: PRESENT + EXECUTED (mechanically verified) — **not** INTEGRATED as an improvement (on the env it was meant to fix, the mechanism it enables is harmful).
- Progress guard: PRESENT + EXECUTED, provably inert where shaping exists (bit-identity) — INTEGRATED nowhere.
- ECR itself: REPRODUCED (EXP-FP-0080) on delayed_reward; GENERALIZING on the corridor axis (→ delayed_multistep, EXP-FP-0125). No maturity change.

## Provenance

- Preregs: `experiments/preregistration_EXP-FP-0126.json`, `experiments/preregistration_EXP-FP-0127.json` (sealed before implementation/runs)
- Module: `prototypes/architecture-a/attention_ecr_detector.py` (additive; canonical ECR module untouched)
- Drivers: `prototypes/architecture-a/experiments/exp_fp_0126_ecr_detector.py`, `exp_fp_0127_ecr_guard.py`, `write_receipt_0126_0127.py`
- Tests: `prototypes/architecture-a/tests/test_ecr_detector.py` (15/15)
- Receipts: `receipts/EXP-FP-0126.json`, `receipts/EXP-FP-0127.json` (hash-chained; chain positions clean)
- IDs: EXP-FP-0126, EXP-FP-0127 claimed via `experiments/id_registry.py`, lane ecr-generalization, family EXP-FP-012x
- Seeds: 0126 grid_world {88101–88104}, delayed_reward {88201–88204}; 0127 grid_world {88301–88304} — all fresh at sealing
- Protocol note: the first 0126 run was declared VOID on a frozen gate (G2 driver bug — tuple unpack; executor bug, no data interpreted), fixed code-only, rerun deterministically with G2 exact-match
- CONSCIOUSNESS: UNRESOLVED — closed-loop returns + gain telemetry only
