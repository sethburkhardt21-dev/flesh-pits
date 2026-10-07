# Race-vendored consolidation implementations — PROVENANCE

Byte-identical copies (sha256-verified, read-only reference) of the two
honest batch consolidation implementations raced by
`consolidation_race.py` (EXP-FP-0005, §9 item 8 — the audit's
KEEP_AND_DEEPEN race). The originals were NOT modified; these copies
exist so the race runs inside `flesh-pits/` (standing order: work
solely in the Flesh Pits tree) without import-time drift.

| copy | original source | sha256 |
|---|---|---|
| `consolidation_cycle_s01.py` | `flesh-pits/var/scratch-mneumora-main/mneumora/core/malice_port/consolidation_cycle.py` (MALICE M1, S-01) | `c75e3892b90973c303ad47c5c87126c456ac3a61d44c18a987816f4fdb58db30` |
| `consolidation_cycle_memory_port.py` | `flesh-pits/var/scratch-mneumora-main/mneumora/core/memory_port/consolidation_cycle.py` | `4f6707cc102310e00452a38e203633ca5be263209ff60b68ad5d3101e95bd9da` |

Verified 2026-10-07: `sha256sum` of copies == `sha256sum` of originals
(exact match, recorded above). Class names collide (`ConsolidationCycle`
in both) — the race module imports them under aliases and never renames
the originals. Do not edit these files; if the sources change, re-vendor
and record the new hashes.
