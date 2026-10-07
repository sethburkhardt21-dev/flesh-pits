# Flesh Pits operator checklist — standing gates

*Every run, every push. A gate skipped is a claim unearned.*

## Before running experiments
- [ ] Preregistration written (hypothesis, null, metric, baseline, seeds) — §38
- [ ] Config hashed; run ID minted; backend rung recorded (runtime_ladder.json — a backend change ends the run)
- [ ] Measurement code tripwire-clean: `python3 ~/workspace/skills/fabrication-tripwire/fabrication_tripwire.py <dirs>` — any `random.*` in executable code fails the run

## Before declaring a result
- [ ] Receipt written via the harness (hash-chained — `verify_chain()` passes)
- [ ] Kill/lesion attempted against your favorite claim (§40)
- [ ] Negatives appended to research/negative_results.md
- [ ] Maturity level claimed ≤ evidence (PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED)
- [ ] CONSCIOUSNESS: UNRESOLVED — mechanisms, never narratives

## Before pushing to GitHub
- [ ] Re-stage: rsync excludes `var/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.git/` — never `--delete` (stage holds README.md, .gitignore)
- [ ] Export hygiene: leak-scan clean; pii-redaction findings verified (seeds/hashes are known false positives — verify before redacting); chamber-16 sweep clean
- [ ] Tests green: flesh-pits experiments suite + prototype suites
- [ ] Brothel tree untouched (verify hash if in doubt); var/ never pushed
- [ ] Push via `~/workspace/push_flesh_pits.py`; read back remote HEAD and match

## Standing prohibitions
- NEVER touch `~/workspace/mneumora` (live Being) or `~/workspace/chambers/sleeping-quarters` (chamber 16)
- NEVER work, modify, or push the brothel tree (dormant)
- No safeguard/credential/auth bypass work, ever
- 429-class errors = hard stop for that provider scope
