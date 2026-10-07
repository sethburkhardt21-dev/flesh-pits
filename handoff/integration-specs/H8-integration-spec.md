# H8 — wire-by-decision-receipt discipline — integration specification

**Package:** `flesh-pits/handoff/packages/H8` (src sha256 in `manifest.json`)
**Maturity:** EXECUTED. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface

This is a **method, not a mechanism** — it governs HOW any lab finding
(H1–H6 or future) gets wired into a tick path, primary or lab. The
integration surface is the primary's **wiring/change procedure**: adopt
the five-gate discipline as the mandatory checklist before any mechanism
lands in the live tick.

No code module is assumed. The "adapter" is a procedure the primary lane
executes, producing one hash-chained decision receipt per wiring
(the H7 harness's `write_receipt` is the natural vehicle).

## 2. Interface contract (the five gates)

Reference implementation: `src/k8_wiring_method_reference.py` (the actual
K8 script, renamed; sha256 `892e06bf...`). Canonical evidence:
`receipts/k8_r1_wiring_decision.json` (WIRE, recorded 2026-10-07T07:23:05Z)
+ `k8_r1_wiring_confirmation.json`.

For a candidate wiring W of mechanism M into tick path T, preregister and
run all five gates. Decision = WIRE iff all pass, else REJECT with the
failing gate(s) stated:

- **Gate a — behavioral identity (frozen):** with the new mechanism in its
  null/frozen configuration, the tick's BEHAVIORAL CONTENT must be
  identical to the pre-wire baseline on a fixed probe set. "Behavioral
  content, not process-global sequence numbers" — the NR-A-010 lesson:
  the first K8 implementation compared raw item_ids and failed on a
  harness artifact; the corrected gate compares behavioral content
  (what consumers received, what actions were taken). Name WHAT is
  compared in the preregistration.
- **Gate b — byte-identical reruns:** every prior experiment in the
  affected area re-runs byte-identically post-wire (K8: K1–K3 reruns,
  sha prefixes e007671877da7e95, bd2260363c15c, 0fc80361213e7c93).
  Any divergence = the wiring changed proven behavior = REJECT.
- **Gate c — non-degradation:** the mechanism's own superiority probe
  (K8: K4 learned/frozen ≥ margin, 4/4) still passes post-wire.
- **Gate d — sole-path:** the wiring introduces no side channel around
  the mechanism's invariants (K8: zero non-feedback consumer deliveries
  on 19–21 sub-ignition trials per seed). Port the invariant to the
  mechanism being wired.
- **Gate e — recovery replication:** a known recovery/failure behavior
  replicates post-wire (K8: freeze recovery replication).

Plus: **post-wire byte-identical reruns of every prior experiment**
(K8 re-ran K1–K4: 4832d984137bfb6d), all recorded in ONE hash-chained
decision receipt. And the null must be live: the procedure must be able
to produce REJECT (the package test demonstrates the REJECT path on a
sabotaged toy wiring — a method that cannot say no is decoration).

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: wiring that cannot silently change behavior. The K8
canonical run: WIRE decision with all gates PASS on fresh seeds;
post-wire K1–K4 reruns byte-identical.

Preregistration (stranger-runnable):
1. **Standalone demonstration:** `cp -r H8 /tmp/pkgtest && python3
   tests/test_h8.py` — five gates WIRE on honest wiring, REJECT on
   sabotage, PASS.
2. **First primary wiring under the discipline:** the primary lane picks
   its first mechanism wiring (H1 is the natural candidate), writes the
   preregistration naming all five gates' exact comparisons, runs them,
   and publishes the decision receipt. Gate: receipt exists, is
   hash-chained (H7), and states WIRE or REJECT with per-gate evidence.
3. **REJECT-path drill:** deliberately sabotage one gate's condition in a
   scratch branch (e.g. break behavioral identity) — gate: the procedure
   returns REJECT naming the failing gate. A discipline that has never
   produced REJECT on the primary is unproven.

## 4. Rollback + tripwire

- **Rollback:** n/a (method — per manifest). If the discipline is
  abandoned, wirings revert to ad-hoc — which is exactly what the
  discipline exists to prevent. The decision receipts already written
  remain valid evidence either way.
- **Tripwire (proves harm):** any wiring merged to the primary's main
  WITHOUT a decision receipt naming all five gates is a discipline
  violation — flag it in review. The method's ceiling is gate quality
  (manifest risk): if gate (a)'s "behavioral content" comparison is
  ever weakened to sequence-number comparison, the NR-A-010 artifact
  class returns — the tripwire is a review checklist item, not code.

## 5. Bounds and risks (carried over verbatim, not softened)

- The method is only as good as the gates — behavioral-identity must
  name WHAT is compared (NR-A-010 lesson: behavioral content, not
  process-global sequence numbers).
- Gate quality is the method's ceiling (per manifest).
- The package test demonstrates the method on a toy wiring; canonical
  evidence is the vendored K8 decision receipt. Do not cite the toy as
  the proof — cite k8_r1_wiring_decision.json.
