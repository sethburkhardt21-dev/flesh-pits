# PROVING GROUND — flesh-pits skill/mechanism evaluation

*Phase 4 coordinator tasking. Worker: proving-ground subagent. Scope:
`~/workspace/chambers/emergent-mind/flesh-pits/` only. brothel/ untouched
(dormant). `~/workspace/mneumora` untouched. sleeping-quarters untouched.
Desktop Commander route dead — all work local. CONSCIOUSNESS: UNRESOLVED.*

*Standing law applied throughout: NOTHING EVER FAKE, ONLY REAL. Every
verdict below carries its measurement. INTEGRATED means the gating test
passed (removing it changes something measurable). REJECTED-as-theater
means the null measurement showed zero difference.*

---

## 1. fabrication-tripwire — INTEGRATED (verification instrument)

**Measurement.** `scan_for_fabricated_numbers` run over every lab code
directory: `experiments/`, `prototypes/`, `ablations/`, `benchmarks/`,
`donors/`, `handoff/`. Result: **0 offenders** across the full tree
(0.07s).

**Non-vacuous proof.** The scan is not passing on empty input: the lab
uses randomness in 9 files (`random.Random(seed)` seeded instances in
`env_interface.py`, architecture-a `envs.py`/k1/k3, architecture-b
`generative_model.py`/`active_inference.py`/`agent.py`, tests). The
tripwire correctly ignores seeded-instance usage by design — that is the
reproducible-experiment pattern, not fabrication.

**Positive control.** Synthetic dirty file (`return random.randint(1,100)`
in executable code) flagged at the exact line; clean file (comment
mention, string literal, seeded `random.Random(42)` + `rng.random()`)
ignored. Control passes.

**Wiring.** Scan proven on live lab code; command documented for the
experiment-run checklist:
`python3 -c "from fabrication_tripwire import scan_for_fabricated_numbers; ..."`
from `~/workspace/skills/fabrication-tripwire/`. Not spliced into
`harness.py` unilaterally — that coupling decision belongs to the
coordinator. The instrument is proven; the permanent gate is a
one-line checklist addition.

---

## 2. code-sandbox — REJECTED-as-theater

**Null measurement.** `donors/` is **empty** — no donor code has entered
the lab. All lab code is lab-original, owner-held, stdlib-only, and
already exercised by the lab's own suites (experiments 24/24 green,
architecture-b 21/21 green). No `eval`/`exec` of foreign code anywhere:
the single `eval()` in the tree
(`experiments/baselines/persistent_no_learning.py:58`) deserializes keys
from the agent's own `_extra_state()` snapshot (first-party
`str(int)`/`str(tuple)` keys) — a hygiene note (`ast.literal_eval`
would be strictly better), not an untrusted-code case. "Ported from
donor" markers (e.g. architecture-a `broadcast.py:10`) are hand-adapted
lab code, not executed foreign code.

Wiring the AST+hardened-process executor around already-trusted,
already-tested first-party code changes nothing measurable. Revisit
trigger: the day a non-empty `donors/` drop or any network-fetched code
enters the experiment path.

---

## 3. file-lock — REJECTED-as-theater

**Null measurement.** Sweep of every file-writing call in lab code:
all receipt/experiment writes are mode `"w"` to unique per-experiment
paths (`write_receipt` → `<experiment_id>.json`; architecture-b
`experiments.py:144`, `:652`; `predictions.py:124` opens one JSONL per
run). **Zero append-mode opens** in the tree. `var/checkpoint.log` holds
5 lines appended sequentially by build units (timestamps 06:15–06:25,
single writer, no interleaving). No parallel workers active; no cron
touches flesh-pits. There is no contested write to guard — wiring the
lock changes nothing measurable.

**Mechanism soundness (positive control).** `FileLock` re-acquire while
held → `False`; `is_locked()` → `True`; after context-manager release →
`False`. The rejection is fit-based, not breakage-based.

**Revisit trigger (documented, not wired):** if parallel experiment
workers ever share an append target (registry, checkpoint log), wrap
those appends in `with FileLock(...)`.

---

## 4. skill-evolution — REJECTED-as-theater

The skill versions *iterating playbooks* (SKILL.md + observations /
inspections / amendments). The lab has no iterating playbook: the
experiment contract (`ENV_INTERFACE.md` v1.0.0) is declared STABLE, the
kill-experiment procedure is fixed, maturity updates are event-driven
(kill verdicts), not iterative. Initializing `.versions/` + JSONL logs
would record observations about a static protocol — bookkeeping theater.
Revisit trigger: if the lab develops a genuinely iterating evaluation
procedure (e.g. the maturity auditor proposed in §9), evolution tracking
attaches to it then.

---

## 5. Memory provenance (ProvenanceStore pattern) — REJECTED-as-theater

**Null measurement (executed).** Called
`ProvenanceStore.record_write_provenance` for a receipt write
(`receipts/EXP-FP-0001-D.json`) and a registry write, origin `agent`.
Both returned **`None` — nothing recorded.** The store's strict path
allowlist admits only `MEMORY.md`, `memory.md`, `USER.md`,
`users/<id>/USER.md`, `memory/*.md`; every lab path is rejected. Wiring
it as-is records literally zero provenance rows.

**LANDING — false premise corrected.** The tasking premise ("the
harnesses already hash-chain receipts") is **inaccurate for the
flesh-pits harness**. Measured: receipts carry per-receipt
`config_hash` + per-episode `init_hash`/`final_hash` (env-state seals),
but **no cross-receipt chain** — no `prev_receipt_hash` field anywhere,
no `verify_chain` function in `experiments/harness.py` (the
`verify_chain()` mention in HARVEST_CANDIDATES.md attributes it to the
*brothel* harness variant). The receipt *set* has no tamper-evident
ordering; a receipt could be added/removed/reordered undetected
(config tampering within one receipt is caught by its `config_hash`;
set-level tampering is not).

**Verdict.** The ProvenanceStore pattern answers a question the lab does
not have (single trusted writer; origin-class tracking for memory
markdown). REJECTED-as-theater in current form. The genuine gap it
exposes — no cross-receipt chain — is recorded as a follow-up for the
coordinator (minimal honest implementation: `prev_receipt_hash` embedded
at write time in `write_receipt`), not built unilaterally here.

---

## 6. Budget enforcement on var/ — INTEGRATED

**The real use case.** `var/` = **153,209,107 bytes** (146 MiB), of which
the frozen reference checkout `scratch-mneumora-main/` = 153,207,599
bytes. Unbounded scratch growth; no prior guard.

**Built** (validate_tree pattern, stdlib only, reader-abstracted walk):
- `research/var_budget.py` — `parse_policy` (versioned, strict),
  `validate_var_tree` (audit), `check_write` / `assert_write_allowed`
  (pre-write gate, raises `BudgetExceeded`), CLI (`check` / `gate`).
  Streaming byte counts; contents never read; symlinks not followed;
  `..` traversal rejected.
- `research/var-budget-policy.json` v1 — `maxTotalBytes` 160 MiB,
  `maxFileBytes` 20 MiB, `maxDepth` 16 (runaway-nesting guard),
  `scratch-mneumora-main/**` subtree cap 154,252,492 B (frozen
  reference must not grow), `checkpoint.log` 1 MiB.
- `research/test_var_budget.py` — **15/15 pass**, including live-tree
  audit and refusal paths.

**Gating-test evidence (live, just ran):**
- `check`: var/ 153,209,107 B vs 160 MiB cap → within budget, no
  violations (exit 0).
- `gate experiments_out/big_dump.bin 52428800` → **REFUSED**
  (per-file cap + projected 205,637,907 B > total cap; exit 1).
- `gate scratch-mneumora-main/new_file.bin 2097152` → **REFUSED**
  (subtree would reach 155,304,751 B > 154,252,492 B cap; exit 1).
- `gate experiments_out/small.bin 1048576` → allowed (exit 0).

Removing this changes something measurable: the next 50 MB accidental
dump lands silently. It stays.

---

## 7. Contradiction resolution — APPLIED as procedure; REJECTED as automation

Applied the resolver's detection rule (same subject + same predicate +
different object) across the lab's recorded beliefs
(registry, negative_results, both MATURITY ledgers, PROTOTYPE_PROMOTION,
ARCHITECTURE_FINDINGS, HARVEST_CANDIDATES, REJECTED_IDEAS).

**GENUINE CONTRADICTION — DISPUTED, escalated to coordinator:**
- F1 `prototypes/architecture-a/MATURITY.md` (06:49): K4 canonical
  re-run is **pending** — "K4 must be RE-RUN against the canonical spec
  when it lands."
- F2 `handoff/ARCHITECTURE_FINDINGS.md` (06:49): K4 canonical re-run
  **passed** — "K4 re-run PASS 4/4 on canonical env."
- Evidence on disk supports neither cleanly: no receipt documents a
  4/4 canonical-env K4 re-run; `repro_k4_attention_baseline.json`
  (06:56, *after* both docs) claims **5/5 seeds** with no env field;
  `k5_multitask_generalization.json` (06:58) mentions no canonical env.
  Per the resolver: cannot resolve mechanically → both stay DISPUTED.
  **Coordinator must rule:** either produce the 4/4 receipt or correct
  one of the two documents. This is exactly the failure mode a
  mechanical consistency check would have caught at write time.

**Owned tensions (not contradictions — the lab already holds them):**
- D as "the bar, honestly weak": resolved in REJECTED_IDEAS.md #11 —
  D is a *characterized floor*, the symbolic baseline is the real bar.
- `'agent'` provenance origin unreachable via public API: documented as
  a design gap in both the findings and harvest docs.
- Rejected ideas vs HARVEST_NOW: cross-checked — all 11 rejections stay
  out of HARVEST_NOW (appear only under REJECT/INCONCLUSIVE).

**Wiring verdict.** The resolver's code operates on structured
`{subject, predicate, object}` facts; the lab's beliefs live in prose
markdown with no fact store. Automating it now would require inventing
the store — theater. The *procedure* is what earned its place (it found
the K4 contradiction); it stays a manual audit step, ideally mechanized
by the auditor proposed in §9.

---

## 8. Evaluation feedback — ASSESSED; path EXISTS; gap PROPOSED

**Assessment: the failure-detection → competence-estimation path works.**
Preregistered kill conditions → measured verdicts → maturity-ledger
updates → harvest triage, all evidenced: B/K4 killed error-affect →
`MATURITY.md` marks it "CAUSAL, REJECTED … Dropped; PAD kept"; B/C2
killed precision → "Do not ship as-is"; B/C4 killed the AI label →
"INTEGRATED, label removed"; all three land in REJECTED_IDEAS.md and
stay out of HARVEST_NOW. The loop is manual (markdown), not computed —
but it is real and it fires.

**The gap is mechanical consistency, not a missing path** — exhibit:
the §7 K4 contradiction (a quantitative claim with no matching receipt,
contradicting a same-day ledger). 

**Minimal honest implementation (proposed, not built — coordinator
decision):** `research/maturity_audit.py` — reads kill receipts +
MATURITY ledgers + HARVEST_CANDIDATES and checks: (1) every
KILL/REJECTED verdict in a receipt has a matching REJECTED ledger
entry; (2) every "PASS n/n"-style quantitative claim in findings docs
has a receipt on disk whose numbers match; (3) no REJECTED component
appears under HARVEST_NOW. Exit non-zero on mismatch. This is the
backstop the manual loop lacks; it would have flagged the K4 4/4 claim
at write time.

---

## 9. leak-scan + pii-redaction — VERIFIED with a documented false-positive pattern

**leak-scan** behaves per spec (verified live): clean prose passes;
`YYYY-MM-DD` dates and `/home/` paths raise `PublicLeakError`. Note for
exports: lab receipts/registry legitimately contain UTC timestamps and
dates — the invariant means exports must strip them, not that the lab
records are dirty.

**pii-redaction — false-positive pattern CONFIRMED on experiment seeds
(measured):**
- `seed=1839661001` (episode seed) → flagged **PHONE, 0.9**
- `6630382182` (10-digit substring of a sha256 config hash) →
  flagged **PHONE, 0.9**

Every lab receipt will fire PHONE findings on seeds and hash
substrings. My assessment matches the coordinator's "clean" stage
verdict **only under a known-FP exclusion**: 9–10 digit numerics in
`seed=`/`episode_id` contexts and inside 64-hex hashes are experiment
artifacts, not phone numbers. Running pii-redaction blind over
`receipts/` without that exclusion is a false-alarm generator — the
exclusion list should travel with any future scan of lab artifacts.

Also confirmed on this very report: the NAME heuristic fires at 0.8 on
every markdown bold-header word ("Phase", "Worker", "Scope", …) — the
documented short-capitalized-word FP pattern, now observed on markdown
formatting itself. Treat all 0.8 NAME findings on formatted prose as
low-trust.

---

## Files created (this run)

- `research/var_budget.py` — byte-budget validator + pre-write gate + CLI
- `research/var-budget-policy.json` — v1 policy (160 MiB total, frozen-subtree cap, depth guard)
- `research/test_var_budget.py` — 15/15 green
- `research/proving_ground.md` — this report

## Escalations for the coordinator

1. **K4 canonical re-run contradiction** (§7): rule between MATURITY.md
   "pending" and FINDINGS.md "PASS 4/4", or produce the missing 4/4
   receipt. The 5/5 repro receipt postdates both claims and names no env.
2. **Receipt-chain gap** (§5): flesh-pits receipts are not cross-chained;
   consider `prev_receipt_hash` in `write_receipt` (harness change —
   your call, not made here).
3. **Maturity auditor proposal** (§8): `research/maturity_audit.py` spec
   above — say the word and it gets built to the same standard as the
   budget guard.
4. **Tripwire permanent gate** (§1): one-line checklist addition to the
   experiment-run procedure; harness modification deferred to you.
5. **PII scan exclusion** (§9): seed/hash-substring FP pattern documented;
   attach the exclusion to any receipt-scanning pipeline.

*No decorative wiring was added. Every INTEGRATED item above refused or
measured something on live lab state; every REJECTED item carries its
null measurement. Landings and failures only.*
