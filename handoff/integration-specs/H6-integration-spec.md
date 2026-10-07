# H6 — memory_provenance hash-chaining — integration specification

**Package:** `flesh-pits/handoff/packages/H6` (src sha256 in `manifest.json`)
**Maturity:** CAUSAL. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

**Standing fact: this is already live on the primary.** The harvest entry
names the live Being tree `core/memory_port/memory_provenance.py`
(already live-wired). The H6 package was built from the lab-local copy
inside flesh-pits and is listed for completeness per §51. This spec
therefore covers the REMAINING SURFACE, not a fresh import.

## 1. Integration surface (the remaining surface; seam from scratch copy)

- **Canonical module:** `core/memory_port/memory_provenance.py` —
  `ProvenanceStore` (path-scoped, hash-chained, retention law,
  compare-and-set rollback, path allowlist). LIVE per the harvest entry;
  the live tree's exact call sites are UNVERIFIED from inside the lab (Q6).
- **Remaining surface — every memory write path must call
  `record_write_provenance`:**
  - `core/memory_integration.py` — `CanonicalMemory.admit(...)`,
    `admit_world_step(...)`, `admit_broadcast_envelope(...)`
    (ASSUMED names from the scratch copy)
  - `core/memory_port/experience_store.py` —
    `ExperienceStore.store_experience(...)` (if it persists to disk)
  - Any scheduler path that writes memory snapshots/checkpoints
    (`CanonicalMemory.checkpoint()` / `restore()` — writes of state
    files are provenance-worthy too)
  - The H5 consolidation summary layer: summary writes should carry
    provenance records (origin_class per the retention law)
- **Coverage test the primary lane should run:** enumerate all
  `open(path, "w")` / `os.replace` write sites under the primary's
  memory directories and assert each one is preceded by (or wrapped with)
  a provenance recording. Any write without provenance is a gap, not a
  judgment call.

## 2. Interface contract (as implemented; do not weaken)

```python
class ProvenanceStore:
    def __init__(self, path: str, path_policy: PathPolicy = DEFAULT_PATH_POLICY)
    def record_write_provenance(self, *, workspace_dir: str, relative_path: str,
                                content_before: str, content_after: str,
                                origin_class: str, observed_at: int,
                                session_id: str | None = None,
                                session_key: str | None = None):
        # RETENTION LAW: 'agent' is kept ONLY when a previous record exists
        # with 'agent' origin AND its file_hash == sha256(content_before).
        # Otherwise — INCLUDING EVERY FIRST WRITE — the record is 'untrusted',
        # no matter what origin_class the caller claims.
        # Returns a rollback closure (compare-and-set on reservation_id),
        # or None if the path is rejected by the allowlist.
        # Raises OverflowError when the store is full (fail closed, not silent).
    def clear_provenance(self, *, workspace_dir, relative_path, content_before)
        # deletes ONLY if the pre-write hash still matches
    def read_provenance(self, *, workspace_dir, relative_path) -> dict | None
    def list_provenance(self, *, workspace_dir) -> list[dict]
```

Hard properties the primary must preserve:
- Bootstrap semantics: a first write can NEVER retain 'agent' (the
  CORRECTED semantics — previous MUST exist). Any caller claiming
  `origin_class="agent"` on a first write gets 'untrusted'. This is the
  anti-forgery core; do not "fix" it into trust-on-first-write.
- Rollback is compare-and-set on `reservation_id`: if someone wrote after
  us, rollback does NOT clobber. Preserve this.
- Path allowlist: `normalize_relative_path` returning None → record
  refused (returns None). Writes outside the allowlist must not gain
  provenance records — but the WRITE ITSELF is the caller's concern;
  provenance refusal is not a write block.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: tamper-evident memory writes — every covered write is
attributable to (workspace, relative path, before/after hashes, origin
class, timestamp), and silent store tampering breaks the chain
detectably. Lab evidence: CAUSAL, tick-fresh in live var/being (Phase-1
audit); package test proves chaining standalone (bootstrap degrades,
retention, tamper-breaks-chain, rollback, allowlist, store-tamper
detection).

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H6 /tmp/pkgtest && python3
   tests/test_h6.py` (see package README for the exact test name) — PASS.
2. **Coverage audit (the main remaining work):** enumerate the primary's
   memory write sites; gate: 100% of write sites under the memory
   directories call `record_write_provenance`, OR the gap is explicitly
   listed with a reason. No silent gaps.
3. **Tamper drill:** modify a provenance store file out-of-band; the next
   `read_provenance`/`_valid` must return None (invalid), and the
   retention law must degrade the next write to 'untrusted'. Gate: the
   drill is scripted and passes.
4. **Bootstrap drill:** delete a provenance record, re-run a write with
   `origin_class="agent"`; gate: the new record is 'untrusted'.
5. **Rollback drill:** record, write again from another session, then
   call the first rollback closure; gate: the second write's record is
   NOT clobbered (compare-and-set holds).

## 4. Rollback + tripwire

- **Rollback:** n/a — already live (per manifest). If a write path must
  be exempted, the exemption is a named, reviewed allowlist entry, not a
  silent skip. Full removal would be deleting the call sites — the
  primary lane owns that decision; this spec does not recommend it.
- **Tripwire (proves harm):** a scheduled audit (the Phase-1 audit pattern)
  that re-runs the tamper + bootstrap drills. If `read_provenance` starts
  returning records that fail `_valid` (chain broken but reads succeeding),
  or if any 'agent'-origin record appears without a valid predecessor
  (retention-law violation), the store's integrity assumptions are broken —
  quarantine the store file, alert, do not write new 'agent' records until
  the cause is found.

## 5. Bounds and risks (carried over verbatim, not softened)

- 'agent' provenance genesis path is unreachable via the public API
  (design gap — INCONCLUSIVE item; the test seeds the genesis record to
  exercise the retention rule). **This is the open design question:**
  how does a legitimate first 'agent' write ever happen? Until the
  primary's architect answers it, every first write is 'untrusted' by
  law — which is honest but means 'agent' provenance can currently only
  continue, never begin. See Q6 in OPEN_QUESTIONS.md.
- Package built from the lab-local copy, not the live tree; canonical
  source remains the live tree. If the live tree's module has drifted
  from the packaged copy (sha256 `bff6f99c...`), diff before trusting
  this spec's API quotes.
