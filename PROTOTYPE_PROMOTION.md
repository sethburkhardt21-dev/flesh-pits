# Prototype promotion — brothel → flesh-pits (2026-10-07)

**Standing owner order:** flesh-pits only, 24/7. The brothel tree is dormant —
untouched on disk, not deleted, not worked, not pushed.

**Action taken:** `brothel/prototypes/architecture-a/` and `brothel/prototypes/architecture-b/`
were COPIED (not moved) to `flesh-pits/prototypes/architecture-a/` and
`flesh-pits/prototypes/architecture-b/`. All further architecture work happens on
the flesh-pits copies.

**Dormancy proof:** brothel tree file-hash (sorted sha256sum over all files)
identical before and after the copy:
`0d9d3513156b7c5eb0f8b93436b818cd7fe5078af1168a41e677fa3697903af4`

**Migration check (flesh-pits copies, 2026-10-07):**
- architecture-a K2 broadcast lesion: PASS (selective + total + recovery, no side channels)
- architecture-b unit tests: 21/21 green

**Handoff docs** (ARCHITECTURE_FINDINGS.md, HARVEST_CANDIDATES.md, REJECTED_IDEAS.md)
copied from top-level handoff/ into flesh-pits/handoff/ for the GitHub stage.
Top-level handoff/ remains the canonical location.
