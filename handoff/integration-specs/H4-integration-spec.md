# H4 — change_bid (RunningZScoreBid) — integration specification

**Package:** `flesh-pits/handoff/packages/H4` (src sha256 in `manifest.json`)
**Maturity:** CAUSAL. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

**Key finding for the primary lane: the mechanism already exists on the
primary.** The scratch copy shows two audit twins of this exact mechanism:

- `core/self_model_port/change_bid.py` — `RunningZScoreBid` (same class name,
  same `bid`/`baseline` API)
- `core/malice_port/habituation_bid.py` — the donor original

The H4 package is the lab-portable, standalone-verified reference of that
same mechanism (sigmoid of a causal running z-score; change-detection and
habituation-desaturation as ONE mechanism). The integration decision is
therefore a **deduplication decision**, not a fresh import:

- **Option A (recommended):** keep the primary's existing twin(s) as
  canonical; use H4's package as the portable reference + test suite for
  the primary's implementation (port `tests/test_h4.py`'s
  shift/habituation/fail-closed/entropy-collapse probes against the
  primary's `RunningZScoreBid`).
- **Option B:** if the primary's twins diverge in behavior, the H4 vendored
  `bids.py` (sha256 `292dc50e...`, byte-identical to H1's dependency) is
  the proven reference to converge toward.
- **Option C:** if the primary has NO novelty/salience input at a given
  selection point, wire H4's `RunningZScoreBid` as that point's bid source —
  the ASSUMED seam is wherever raw thresholds currently feed selection
  (e.g. upstream of H1's arbitrator, or any novelty input on the primary).

Live-tree status of the twins is UNVERIFIED (scratch copy only — Q4).

## 2. Interface contract

Vendored: `src/bids.py` — `RunningZScoreBid`, `NonFiniteBidRefused`,
`stable_sigmoid`, constants `EMA_ALPHA=0.01`, `SD_FLOOR=1e-9`.

```python
class RunningZScoreBid:
    def __init__(self, alpha: float = 0.01, initial_var: float = 1.0)
    def bid(self, value: float) -> float:
        # sigmoid((x - running mean) / running sd); statistics READ BEFORE update
        # first reading bids 0.5; sd <= 1e-9 -> z = 0 (constant input never amplified)
        # sustained level shift: baseline follows at rate alpha, bids relax to 0.5
        # raises NonFiniteBidRefused on non-finite input — STATE UNCHANGED
    @property
    def baseline(self) -> (mean, var, seen)
```

Semantics the adapter must preserve:
- **Causal:** the bid for tick t uses statistics from ticks < t only.
- **Habituation built in:** a constant input bids 0.5 forever (no
  saturation) — do not add a separate habituation stage on top; the
  manifest is explicit that these are one mechanism.
- **Fail-closed:** non-finite input raises before any state mutates.
- **State ownership:** `(mean, var, seen)` per bid instance; persist with
  the owning module's state.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: a cheap honest novelty signal — zero-latency shift
detection (lab battery: drift 22 vs raw-threshold 69). CAUSAL in two
independent tests; CAUSAL for moment-to-moment arbitration under frozen
gains (lab A: K1/K4 loop).

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H4 /tmp/pkgtest && python3
   tests/test_h4.py` — shift/habituation/fail-closed/entropy-collapse, PASS.
2. **Twin-equivalence probe (Option A):** run the H4 package test's probes
   against the primary's existing `RunningZScoreBid` (both twins). Gate:
   identical bid sequences on a fixed stimulus tape including a level
   shift, a constant run, and a non-finite injection (must raise).
   Divergence → converge on the H4 reference (Option B).
3. **Shift-detection superiority (Option C):** on a synthetic step-change
   tape, detection latency of the z-score bid vs the primary's current raw
   threshold. Gate: strictly lower mean detection latency at matched
   false-alarm rate (lab margin: drift 22 vs 69).
4. **K7-port lesion:** replace bids with constant-0.5 stubs at the wired
   point; arbitration winner-entropy must collapse 1.03–1.38 → ~0.0 bits
   on an alternation probe — proves the bids are load-bearing at THAT
   point. If rewards do NOT move (the NR-A-009 pattern), the bids are
   decoration there — do not claim otherwise.

## 4. Rollback + tripwire

- **Rollback:** trivial — raw thresholds (per manifest). One flag per
  wired point.
- **Tripwire (proves harm):** if winner-entropy collapses to ~0 with a
  static winner while the environment is non-stationary (the K7
  signature), the bid baselines have frozen or the input went constant —
  check the input tape before blaming the bid. If bid outputs saturate
  at 1.0/0.0 persistently, `initial_var`/`alpha` are mismatched to the
  input scale — recalibrate the input, not the constants.

## 5. Bounds and risks (carried over verbatim, not softened)

- **NR-A-009 bound:** the learned system barely needs bids — learned
  gains compensate for dead bids. CAUSAL for moment-to-moment arbitration
  under frozen gains, but do NOT claim bids drive the K4 reward win.
- None identified beyond the NR-A-009 bound (per manifest).
- Note: the primary's existing twins mean H4's marginal value is
  verification/deduplication, not new capability — size the pull
  accordingly (see ORDERING.md).
