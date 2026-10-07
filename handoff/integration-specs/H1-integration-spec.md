# H1 — learned-gain attention loop — integration specification

**Package:** `flesh-pits/handoff/packages/H1` (src sha256 in `manifest.json`)
**Maturity:** GENERALIZING. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

Seam identity comes from the lab-local scratch copy of the primary
(`flesh-pits/var/scratch-mneumora-main/mneumora/`, NOT the live tree — the
scratch copy may be stale; the primary lane must confirm every name below):

- **Primary module:** `core/learned_attention.py`
  - `LearnedAttentionModel.predict_gain(signal_id)` — current fixed-rule gain path
  - `LearnedAttentionModel.observe_outcome(receipt, attended, ...)` — post-hoc feedback path
  - `AttentionGuidedSelector.select(signal_ids, ...)` — the selection/competition point
- **ASSUMED seam:** inside `AttentionGuidedSelector.select()`, replace/augment the
  fixed gain term with `AttentionArbitrator` competition:
  `raw_bid(signal) × learned_gain[signal] → argmax` (label-agnostic tie-break).
  The gain update is written ONLY from `observe_outcome` feedback (mirroring the
  H1 closure: `update_gains()` called solely from broadcast/consumer feedback, never
  a side channel).

**Alternative seam (fallback, UNVERIFIED):** if the primary's selection path is
better described by `core/attention_schema.py` (name seen in scratch copy), the
same arbitrator class attaches there; the contract in §2 is seam-agnostic.

Do NOT invent primary internals: if `AttentionGuidedSelector` does not exist in
the live tree, this spec is BLOCKED on the seam question listed in
`OPEN_QUESTIONS.md` (Q1).

## 2. Interface contract

Vendored files (exact, per `H1/manifest.json`):
`src/attention.py` (`AttentionArbitrator`, `ArbitrationRefused`),
`src/bids.py` (`RunningZScoreBid`, `NonFiniteBidRefused`).
Dependencies also vendored: `broadcast.py`, `workspace_buffer.py`, `ignition.py`
(needed only if the full A-tick is reused; the arbitrator alone needs only
`bids.py`).

```python
class AttentionArbitrator:
    def __init__(self, channels: list[str], *,
                 alpha=0.01, initial_var=1.0,
                 gain_lr=0.1, frozen=False, gain_cap=2.0)
    def arbitrate(self, stimuli: dict[str, float]) -> dict:
        # returns {"winner", "margin", "raw_bids", "competed_bids", "gains", "cycle"}
        # raises ArbitrationRefused on channel mismatch or non-finite stimulus
    def update_gains(self, utility: dict[str, float], reward_baseline: float = 0.0) -> dict:
        # delta-rule: ONLY channels present in `utility` are updated;
        #   gain += lr * (u - baseline) * gain, floored 0.01, capped at gain_cap
        # no-op when frozen (returns current gains); appends (cycle, before, after)
        #   to self.gain_history — the learning evidence trail
    def freeze(self) / def unfreeze(self)
```

Adapter responsibilities on the primary side (primary lane writes this):
- **Stimulus mapping:** convert the primary's per-signal salience/novelty vector
  into `stimuli: dict[channel → finite float]`. Channels are opaque labels
  (identity-symmetric by construction — no privileged identities).
- **Utility feedback:** convert `observe_outcome` feedback into
  `utility: dict[channel → float]` restricted to OBSERVED channels only
  (unobserved channels must be absent from the dict — the mechanism
  deliberately does not punish unobserved arms).
- **State ownership:** the arbitrator holds `gains`, `gain_history`, per-channel
  bid baselines. The primary owns channel set identity and persistence
  (serialize `gains` + bid baselines across restart; format is primary's choice).
- **Error behavior:** `ArbitrationRefused`/`NonFiniteBidRefused` → fail closed
  (hold the previous selection, receipt the refusal). Non-finite stimulus is
  refused before any state mutates.

Minimal-footprint option: ship ONLY `bids.py` + `attention.py` (2 files, stdlib,
~8 KB). The full tick (`tick.py`, `consumers.py`, `broadcast.py`,
`workspace_buffer.py`, `ignition.py`) is available if the primary wants the
complete Architecture-A loop, but that is a separate, larger decision —
recommend the arbitrator alone first.

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta (from lab evidence): adaptive prioritization that tracks
salience-orthogonal relevance shifts — learned-gain / frozen-gain total-reward
ratios 1.61/2.03/1.95/1.57 on the changing-rule env (4/4 seeds, preregistered
margin 1.30); generalization to noisy-signal tracking 1.31–1.52 and
multi-reversal stationary shifts 1.74–2.05.

Preregistration (primary lane executes, stranger-runnable):
1. **Reproduce the lab effect standalone** (no primary needed):
   `cp -r H1 /tmp/pkgtest && cd /tmp/pkgtest && python3 tests/test_h1.py` —
   must PASS and reproduce 1.61/2.03/1.95/1.57 exactly (vendored receipts/
   carries the package test's own receipt). Gate: bit-exact reproduction.
2. **Frozen-mode identity gate (pre-wire, on the primary):** with
   `frozen=True`, the primary's selection behavior with the adapter must be
   byte-identical to the pre-integration baseline on a fixed probe set
   (primary names the probe; frozen gains are exactly 1.0 → competed bids
   equal raw bids; any behavioral delta at frozen mode = adapter bug, REJECT).
3. **Learned-mode superiority (post-wire):** primary-local changing-rule
   probe (primary implements; lab's `tests/envs.py` + `H1b/tests/support/`
   env contract is the reference): learned/frozen total-reward ratio
   ≥ 1.30 on 4/4 fixed seeds (seeds: primary's choice, preregistered before
   the run; lab seeds were the K4 set). Gate: all four ≥ 1.30.
4. **Identity-symmetry spot check:** two novel channel labels with identical
   observation streams must produce bit-identical trajectories (lab's
   symmetry check; replicate in 20 lines).
5. **Non-degradation on stationary tasks:** on the primary's existing
   selection benchmark, learned-mode mean must be ≥ frozen-mode mean − noise
   (primary defines the noise band; 2-sigma of the frozen baseline is the
   suggested gate).

## 4. Rollback + tripwire

- **Rollback (one flag):** `arbitrator.freeze()` — the K4 frozen mode pins
  gains and reverts to the fixed baseline, byte-behavior of the raw-bid path
  preserved. Full removal: delete the adapter call, keep raw thresholds
  (the H1b manifest calls this "trivial — raw thresholds").
- **Tripwire (proves harm):** continuous K4-ratio monitor on the primary's
  own probe: if learned/frozen total-reward ratio < 1.0 for two consecutive
  scheduled runs, OR if arbitration winner-entropy collapses to ~0 bits with
  a static winner (the K7 signature of a dead loop), auto-`freeze()` and
  alert the primary lane. Do NOT tune `gain_lr`/`gain_cap` to escape the
  tripwire — that is the R2 mistake (see H3 spec).

## 5. Bounds and risks (carried over verbatim, not softened)

- **NR-A-006 (STRUCTURAL):** on sparse-delayed-reward tasks the learned
  loop LOSES, R 0.08–0.17; the K9 redesign failed 0/4 (NR-A-011).
  **Never deploy on sparse-delayed-reward tasks.**
- **NR-A-007-without-cue-input:** cue-conditioned changing_rule without the
  cue in the input gives R 1.03–1.09 (architectural input bound) — LIFTED by
  H1b when the cue is in the input. Without H1b, do not deploy on
  cue-conditioned tasks.
- **NR-A-009:** the K4 win is carried by the gain loop, not the bids —
  learned gains compensate for dead bids (K7 lesion collapses arbitration
  entropy 1.03–1.38 → 0.0 bits, yet rewards hold). Do not claim bids drive
  the reward win.
- **NR-A-005:** frozen change-bids suffice on signal-tracking reversals —
  the learned loop is not always load-bearing; deploy it where the probe
  proves a gap.
