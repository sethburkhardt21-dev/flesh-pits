# H1b — cue-indexed gain adapter — integration specification

**Package:** `flesh-pits/handoff/packages/H1b` (src sha256 in `manifest.json`)
**Maturity:** CAUSAL. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

**Depends on H1 landing first** (or landing together as one pull — see
`ORDERING.md`). The adapter is a subclass of H1's `AttentionArbitrator`;
it cannot land without the H1 arbitrator contract.

- **Primary module:** `core/learned_attention.py` —
  `AttentionGuidedSelector.select(signal_ids, ...)` (same ASSUMED seam as H1).
- **Seam:** the primary must expose an observable context at selection time:
  `context_fn(observation) -> hashable`. The tick calls
  `arbitrator.set_context(context_fn(observation))` before `arbitrate()`.
  Without a context signal in the input, this package has no surface —
  the NR-A-007 bound stands (see §5).

Verified against the scratch copy as H1-equivalent naming; live-tree
names UNVERIFIED (Q1 in `OPEN_QUESTIONS.md`).

## 2. Interface contract

Vendored file: `src/attention_cue.py` (`CueIndexedArbitrator`), plus H1's
`attention.py`/`bids.py` (dependency, byte-identical sha256 to H1's).

```python
class CueIndexedArbitrator(AttentionArbitrator):   # attention.py UNTOUCHED
    def set_context(self, context) -> None         # must be hashable; TypeError -> ValueError
    @property
    def current_context(self)
    @property
    def gains(self) -> dict[str, float]            # CURRENT context's vector (lazily 1.0)
    def all_context_gains(self) -> dict           # inspectable: every context's vector (evidence only)
```

Behavioral contract:
- `update_gains()` uses the ORIGINAL delta rule, unchanged; only the active
  context's vector is updated.
- With constant (or absent) context, behavior is EXACTLY the base class
  (the K10 control probe) — any gap there is a confound, not a win.
- `context_fn` missing when the adapter expects it → fail closed
  (per manifest: "context_fn without set_context fails closed").

Adapter responsibilities (primary lane):
- **Context extraction:** `context_fn(observation)` must return a hashable
  (int/str/tuple). Tested only for a binary cue — dimensionality beyond
  binary is UNTESTED (carry this bound).
- **State ownership:** `_context_gains: {context → vector}`. Persist
  per-context vectors (dict-of-dicts JSON is sufficient). Unbounded context
  spaces need a cap + eviction policy the primary defines (e.g. LRU over
  contexts); the adapter does not cap by itself.
- **Error behavior:** unhashable context → `ValueError` before arbitration;
  fail closed (hold previous selection, receipt the refusal).

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: lifts H1's NR-A-007 bound on context-conditioned tasks.
Lab evidence (K10): B1 (cue in input) R = 1.52/1.68/1.64/1.58, 4/4 WIN
(learned 368–392/480 vs frozen 230–244/480); B2 control (cue withheld)
R 0.95–1.13, 0/4 gaps — the cue, not the new module, is causal.
Per-cue gain vectors diverge as designed (e.g. cue0 {a0:1.925, a1:0.875},
cue1 {a0:0.95, a1:2.0}).

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H1b /tmp/pkgtest && python3
   tests/test_h1b.py` — must reproduce B1 1.52/1.68/1.64/1.58 and B2
   0.95–1.13 exactly. Gate: bit-exact.
2. **Constant-context identity:** with `context_fn = lambda obs: None`,
   the adapter's trajectories on the primary's probe must be byte-identical
   to the base H1 arbitrator's (this is the K10 control probe ported).
   Gate: identical episode returns AND identical final gain vectors.
3. **Cue-conditional superiority:** primary-local cue-conditioned probe
   (primary implements; lab reference `tests/support/changing_rule.py` +
   `env_interface.py`): cue-indexed learned / frozen ratios ≥ 1.30 on 4/4
   preregistered seeds; cue-withheld control must show NO gap (ratio
   0.95–1.13 band) — otherwise the gain is confounded.
4. **Per-context inspection:** `all_context_gains()` after the run must
   show divergent vectors per cue (opposite-arm maxima), not one shared
   vector — proves the adapter, not the base loop, carries the win.

## 4. Rollback + tripwire

- **Rollback (one flag):** default `arbitrator_cls=AttentionArbitrator` /
  `context_fn=None` — the tick extension is additive with defaults equal to
  the proven H1 behavior. Reverting the two defaults restores pre-H1b
  behavior exactly.
- **Tripwire (proves harm):** if the cue-conditioned learned/frozen ratio
  drops below 1.0 while the base H1 loop's ratio holds ≥ 1.30 on the same
  probe, the context path is harming arbitration — revert to base class and
  investigate `context_fn` leakage (the classic confound: the context leaks
  the answer). Also watch context-cardinality: if `_context_gains` grows
  unbounded (context space larger than binary), cap it before it becomes a
  memory leak — alert at > 64 contexts.

## 5. Bounds and risks (carried over verbatim, not softened)

- Cue must be IN the input — without it the NR-A-007 bound stands
  (R 1.03–1.09; the B2 control is the proof).
- Context dimensionality untested beyond binary cue. No claim about
  high-cardinality or continuous contexts.
- Single experiment family; second-lane replication open (CAUSAL, not
  GENERALIZING).
