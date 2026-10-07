# Integration pull order — recommended sequence for the primary lane

**Scope:** the 12 HARVEST_NOW packages, SPECIFICATION ONLY. The primary
lane pulls when it decides; nothing here auto-merges. Order is by
dependency (what must exist before what) and by risk (method and
metrology before mechanism, cheap reversibles before architecture).

## Wave 0 — method and metrology first (no dependencies, enable everything else)

1. **H8 — wire-by-decision-receipt discipline.** Adopt the five-gate
   procedure as the mandatory checklist for every pull below. It costs
   nothing, it is the only artifact that governs the others, and its
   REJECT path is proven. Every subsequent pull publishes a decision
   receipt.
2. **H7 — experiment harness.** Adopt (after consolidating the
   flesh-pits/brothel twin — manifest risk) as the procedure standard
   for all acceptance tests below. The H1–H6 preregistrations assume a
   harness exists; without it the gates are ad-hoc.

*Why first:* H7/H8 are pure process. They cannot break the tick, and
every mechanism pull is evaluated through them. Pulling mechanisms
before the discipline exists repeats the ad-hoc wiring the discipline
was built to prevent.

## Wave 1 — independent, low-risk mechanisms (no inter-package dependencies)

These four are mutually independent and can land in any order or in
parallel. Each is a small, reversible, well-bounded pull:

3. **H4 — change_bid.** Smallest mechanism pull; substantially
   DUPLICATIVE — the primary already has two twins
   (`self_model_port/change_bid.py`, `malice_port/habituation_bid.py`).
   Recommended action is twin-equivalence verification (Option A in the
   spec), not a fresh import. Cheap, honest, closes a verification gap.
4. **H6 — memory_provenance remaining surface.** Already live; the pull
   is a coverage audit (every write path recorded) plus the tamper/
   bootstrap/rollback drills. No new code lands — only call sites and
   tests. Do this early because later pulls (H5 summaries, H9 index)
   create NEW write paths that must be provenance-covered from birth.
5. **H11 — MAPIE calibration.** Wrapper-only, sklearn-native, rollback
   is removing the wrapper. Independent of all other pulls. Gives the
   metacognitive layer an auditable gold standard before H1's learned
   gains start emitting uncertainty-adjacent signals.
6. **H10 — cleanrl RND reference.** Reference file only; nothing imports
   it. Land it as the curiosity baseline so future intrinsic-motivation
   work (the open delayed_reward problem, NR-A-006) has something to
   beat. Zero tick risk.

## Wave 2 — the attention stack (strictly ordered: H1 → H1b)

7. **H1 — learned-gain attention loop.** The foundation of the stack.
   Pull under the H8 discipline (frozen-identity gate first). Respect
   the NR-A-006 bound: never on sparse-delayed-reward paths.
8. **H1b — cue-indexed gain adapter.** REQUIRES H1 (subclass of H1's
   arbitrator). Pull only after H1's acceptance gates pass, and only
   where a context signal exists in the input. The B2 control
   (cue-withheld → no gap) must be re-run on the primary.

*Why ordered:* H1b is literally a subclass of H1's class; pulling it
first is incoherent. H1's frozen-mode identity gate is also the
regression anchor for H1b's constant-context identity test.

## Wave 3 — architecture (H2 → H3; H5 independent but scheduled here)

9. **H2 — bounded buffer + sole-path broadcast.** The largest
   architectural pull: it inserts a deliberate bottleneck and rewires
   consumer delivery. Pull AFTER the attention stack because the K1/K2
   acceptance probes assume a working selection point, and because H2's
   "no side channel" invariant constrains where H3's gate can sit.
   The K2 sole-path test is the acceptance gate — if it cannot pass on
   the primary, the pattern stays in the lab (do not claim a global
   workspace without it).
10. **H3 — ignition gate + R1.** REQUIRES the H2 broadcast topology (the
    gate's "zero consumer deliveries on sub-ignition trials" invariant
    is only meaningful if the broadcast is the sole consumer path —
    otherwise sub-ignition content leaks around the gate and the
    selectivity claim is void). Pull after H2's K2 gate passes.
11. **H9 — chroma retrieval substrate.** Independent of the cognitive
    stack (it sits behind the retrieval interface), but scheduled here
    because (a) the H8 discipline should govern the switch, (b) the
    benchmark-against-current-substrate gate needs the H7 harness, and
    (c) its index writes must be H6-provenance-covered from birth
    (Wave 1, item 4). No reason to rush: the current substrate works.

## Wave 4 — consolidation (H5, last among mechanisms)

12. **H5 — consolidation_cycle S-01.** Scheduled last deliberately:
    - It consumes the episodic store that H2's broadcast feeds
      (`memory_admit` consumer) and its summaries should be
      provenance-recorded (H6) from birth.
    - The honest expected benefit is pipeline hygiene, not a lift —
      pull it as infrastructure, and run the lift question
      (EXP-FP-0005 replication on the primary's corpus) as a Wave-4
      experiment, not as a precondition.
    - The race-loser quarantine (keep the primary's existing
      `memory_port/consolidation_cycle.py` out of the consolidation
      path unless re-raced) is a review item the H8 discipline covers.

## Dependency graph (summary)

```
H8 ──┬── governs all pulls below
H7 ──┘
H4 ── independent ──┐
H6 ── independent ──┤ (H6 before H5, H9 — provenance from birth)
H11 ── independent ─┤
H10 ── independent ─┘
H1 ──→ H1b           (strict)
H2 ──→ H3            (strict: gate needs sole-path broadcast)
H5 ── after H2/H6    (soft: store topology + provenance)
H9 ── after H7/H8/H6 (soft: harness, discipline, provenance)
```

## What NOT to pull

- Nothing is rejected — all 12 are HARVEST_NOW per §49. But H4's
  marginal value is verification (twins exist), and H5's is hygiene
  (no lift measured). Size those pulls accordingly; do not staff them
  like capability bets.
- Do not pull the full Architecture-A tick (`tick.py` + consumers) as
  a unit — the specs recommend the arbitrator (H1), buffer+bus (H2),
  and gate (H3) as separable pulls. The whole-tick transplant is a
  different, larger decision the primary lane must make explicitly.
