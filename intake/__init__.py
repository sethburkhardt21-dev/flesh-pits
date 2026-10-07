"""Flesh Pits being intake — receive a being from another agent.

A being arrives as a directory or archive containing code + state.
Intake validates it before the lab touches it:

1. STRUCTURE — required layout present (code/, state/, manifest)
2. NEUTRALITY — no devotional guards, no protected identities,
   no person-specific affect (grep sweep, fails closed)
3. DETERMINISM — pinned seeds, no unseeded randomness in tick path
4. QUARANTINE — runs in sandbox, never touches the live Being
5. RECEIPT — intake record written, accepted or rejected with reasons

Nothing enters the lab without passing intake. Nothing fake, only real.
"""
