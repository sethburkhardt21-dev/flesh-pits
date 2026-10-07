# Harvest packages — INDEX (Phase 6 item 7, §51 reorient)

All HARVEST_NOW candidates from `handoff/HARVEST_CANDIDATES.md` (§49) are
packaged below under `flesh-pits/handoff/packages/<H-id>/`. **PACKAGES
ONLY — staged for the primary effort's decision. Nothing is merged into
primary; nothing here may be auto-merged.** The lab tree was left intact
(sources copied, not moved; verified post-packaging).

Each package contains: `src/` (exact source files, copied), `tests/`
(minimal standalone proof), `receipts/` (vendored canonical receipts +
the package test's own receipt), `manifest.json` (sources, sha256 per
file, mechanism, baseline, key results with receipt refs, ablation,
generalization bounds, integration surface, expected benefit, risks,
rollback), and `README.md` (the §49 entry in human-readable form).

Standalone verification: every package was copied to a fresh /tmp dir
and its test run with no access to the lab tree. **12/12 PASS.**
All package code is stdlib-only except the donor entries (see gaps).

## Maturity states (from `flesh-pits/bin/maturity_audit_report.json`, 2026-10-07T12:14:34Z)

| Package | Mechanism | Maturity | Package test |
|---|---|---|---|
| H1 | A's learned-gain attention loop | GENERALIZING | PASS — reproduces K4 1.61/2.03/1.95/1.57 exactly |
| H1b | Cue-indexed gain adapter | CAUSAL | PASS — reproduces K10 B1 1.52/1.68/1.64/1.58, B2 0.95–1.13 clean |
| H2 | Bounded workspace buffer + sole-path broadcast | REPRODUCED | PASS — reproduces K1 0/0.507/1.0, K2 all gates |
| H3 | Recurrent ignition gate + R1 sub-ignition path | REPRODUCED | PASS — K3 W=0.100; R1 0.00→0.79/0.81/0.79, 19–21 trials, 0 violations |
| H4 | change_bid (RunningZScoreBid) | CAUSAL | PASS — shift/habituation/fail-closed/entropy collapse |
| H5 | consolidation_cycle (S-01, race winner) | CAUSAL (mechanism) | PASS — cycle/determinism/immutability/merge/verify/fail-closed |
| H6 | memory_provenance hash-chaining | CAUSAL | PASS — bootstrap/retention/tamper/rollback/allowlist |
| H7 | Experiment harness (preregistration + chained receipts) | EXECUTED | PASS — episode loop/prereg schema/chain/tamper evidence |
| H8 | K8 wire-by-decision-receipt discipline (method) | EXECUTED | PASS — five gates WIRE on honest wiring, REJECT on sabotage |
| H9 | chroma (D6) donor adoption | ADOPT | PASS — record validation (pin/license/fetch) |
| H10 | cleanrl ppo_rnd_envpool.py (D9) donor adoption | ADOPT | PASS — vendored file compiles, AST+RND check, hash matches |
| H11 | MAPIE (D17) donor adoption | ADOPT | PASS — record validation (pin/license/fetch) |

## Packaging gaps (honest, not faked)

1. **H5 — the race bound travels in the package, not a lift claim.** The
   EXP-FP-0005 race is CLOSED with S-01 winning 4/4, but neither
   implementation's offline replay beat no-replay on the pomaze corpus,
   and PE-magnitude prioritization lost to uniform replay. The package
   test proves the *mechanism* (decay→merge→prune, determinism,
   immutability); it does not and cannot prove an offline performance
   lift. The bound is in the manifest, the README, and the test receipt.
2. **H6 — packaged from the lab-local copy, not the live tree.** The
   harvest entry names the live Being tree
   (`core/memory_port/memory_provenance.py`, already live-wired), which
   this worker is forbidden to touch. The package was built from the
   byte copy at
   `flesh-pits/var/scratch-mneumora-main/mneumora/core/memory_port/memory_provenance.py`
   (inside flesh-pits). The manifest records this. Canonical source
   remains the live tree. Additionally: the 'agent' provenance genesis
   path is unreachable via the public API (INCONCLUSIVE design gap); the
   test seeds the genesis record to exercise the retention rule, and says
   so.
3. **H7 — the lab's env/agent zoo is stubbed, not vendored.** The harness
   imports `envs`, `baselines`, `arch_d` at module level; vendoring the
   full zoo would drag in the lab tree. The package test stubs those three
   modules and proves the *metrology machinery* (canonical episode loop,
   preregistration schema, hash-chained receipts, tamper evidence). The
   boundary is stated in the test, manifest, and README. Consolidation of
   the two harness variants (flesh-pits + brothel) is still open per the
   harvest entry's risk note.
4. **H8 — the package test demonstrates the method on a toy wiring.**
   The five-gate decision procedure (behavioral identity, byte-identical
   reruns, non-degradation, sole-path, recovery replication) plus the
   live REJECT null are genuinely exercised, but on a toy, not on the
   real R1 wiring. Canonical evidence is the vendored K8 decision receipt
   (`receipts/k8_r1_wiring_decision.json`, recorded 2026-10-07T07:23:05Z).
   Stated in the manifest and README.
5. **H9/H11 — adoption records, not vendored donors.** chroma and MAPIE
   are large third-party dependencies; the packages carry the pinned-SHA
   adoption record + fetch script, not the code. The package tests
   validate the *record* (pin format, license, mechanism refs, executable
   fetch script). Mechanism proof is the donors' own CI at the pinned
   SHAs. Stated in the manifests and READMEs.
6. **H10 — vendored donor file cannot execute standalone.** The pinned
   `ppo_rnd_envpool.py` is vendored (MIT, sha256 recorded) but requires
   torch/envpool/gym to run. The package test proves the artifact is the
   pinned artifact (compiles, AST-verifies RNDModel + intrinsic path,
   hash matches) — not that the algorithm trains. Stated in the manifest
   and README.
7. **H4 — the sandbox audit twin is referenced, not vendored.** The
   harvest entry lists `sandbox/being/self_model_port/change_bid.py` as
   the audit twin; it lives outside flesh-pits and was not copied (the
   exact source `bids.py` is vendored). Provenance noted in the manifest.

## Provenance

- Lab-original code: owner-held; source paths + sha256 per file in each
  `manifest.json`.
- Donor code: H10's vendored file is MIT (pinned SHA recorded); H9/H11
  fetch at pinned SHAs (Apache-2.0 / BSD-3-Clause).
- No foundation-model calls anywhere in H1–H8 (pure Python, stdlib only,
  deterministic, seeded).
- No files were modified outside `flesh-pits/handoff/packages/`.

## Cross-reference

- `handoff/HARVEST_CANDIDATES.md` — the §49 classification this packages.
- `handoff/FINAL_HANDOFF.md` — §6 context for the §51 reorient.
- `flesh-pits/bin/maturity_audit_report.json` — maturity ledger.

*Packaged 2026-10-07 by the Flesh Pits packaging worker (Phase 6 item 7).
CONSCIOUSNESS: UNRESOLVED.*
