# Open questions — seams unresolvable from inside the lab

These are questions ONLY the primary lane can answer. Each blocks or
conditions the spec that cites it. Seam names in the specs come from the
lab-local scratch copy
(`flesh-pits/var/scratch-mneumora-main/mneumora/`), which may be stale
relative to the live tree — the live tree was not touched (hard
boundary) and nothing below is asserted about it.

**Q1 — H1/H1b seam identity.** Does `core/learned_attention.py`
(`LearnedAttentionModel`, `AttentionGuidedSelector.select`) still exist
in the live tree with the scratch-copy API (`predict_gain`,
`observe_outcome`, `propose`, `select`)? If the selection path now lives
in `core/attention_schema.py` (name seen in the scratch copy) or
elsewhere, name the real selection/competition point and its gain/feedback
API. Blocks H1 §1; H1b inherits the block.

**Q2 — H2 seam identity.** Do `core/cognitive_workspace.py`
(`Workspace.select`) and `core/broadcast_integration.py`
(`CanonicalBroadcast.broadcast_concerns`, the five consumer classes)
still exist with the scratch-copy API? What is the primary's CURRENT
workspace model (pressure-based selection was visible; is there already
a capacity bound)? Name every real consumer of the broadcast path —
the H2 registry must be built from the real set, not the lab's six.

**Q3 — H3 seam identity.** What does
`core/self_model_port/ignition_arbitration.py` actually do in the live
tree? Is there an existing ignition/threshold concept H3's gate would
duplicate, complement, or replace? Name the real thresholded
propagation point(s) where selected content crosses into consumer
delivery or action.

**Q4 — H4 twin status.** Are `core/self_model_port/change_bid.py`
(`RunningZScoreBid`) and `core/malice_port/habituation_bid.py` both
live, and are they behaviorally identical to the H4 reference
(`bids.py`, sha256 `292dc50e...`)? If they diverged, which one is
canonical? (Determines whether H4 is Option A/B/C in its spec.)

**Q5 — H5 seam identity.** Do `CanonicalMemory.consolidate()`,
`maybe_consolidate()`, and `replay_batch(k=16)` still exist in
`core/memory_integration.py` with the scratch-copy signatures? Is
`core/memory_port/consolidation_cycle.py` (the race loser) still wired
into any live path — and if so, which one, so the loser quarantine in
the H5 spec can be enforced? Where is the offline consolidation
cadence scheduled from?

**Q6 — H6 live call sites + the genesis gap.** Which write paths in the
live tree currently call `ProvenanceStore.record_write_provenance`, and
which memory write paths do NOT (the coverage audit's starting list)?
And the design question: how does a legitimate FIRST 'agent'-origin
write ever happen, given the retention law degrades every first write
to 'untrusted'? (The 'agent' genesis path is unreachable via the public
API — INCONCLUSIVE item from the package.)

**Q7 — H9 current substrate.** What is the primary's current
episodic/semantic retrieval implementation (the baseline Chroma must
beat)? Name the module(s) behind `retrieve(...)` /
`get_similar_experiences(...)` and the embedding function producing
`context_vector`s, so the benchmark gate in the H9 spec has a real
opponent.

**Q8 — H11 confidence points.** Where does the primary currently emit
uncertainty/confidence outputs (the metacognitive layer)? Name the
regressors or estimators whose intervals MAPIE would wrap, and the
current calibration approach (the baseline MAPIE replaces).

**Q9 — H7/H8 existing equivalents.** Does the primary already have an
experiment/verification runner or a wiring-change checklist that H7/H8
would duplicate or conflict with? (Adopt-vs-converge decision for both.)

**Q10 — Scratch-copy staleness.** How far has the live tree drifted from
the lab's scratch copy? If the drift is large, the ASSUMED seams in all
twelve specs need re-derivation — say so early, before any pull begins.

---

*Track C deliverable note:* 12 spec files + ORDERING.md + this file =
14 documents under `handoff/integration-specs/`. All specs are
SPECIFICATION ONLY; no primary tree was read (live) or modified.
CONSCIOUSNESS: UNRESOLVED. Cognitive neutrality preserved throughout
(no privileged-person machinery in any package or spec).
