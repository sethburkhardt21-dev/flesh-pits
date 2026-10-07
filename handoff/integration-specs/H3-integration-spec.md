# H3 — recurrent ignition gate + R1 sub-ignition path — integration specification

**Package:** `flesh-pits/handoff/packages/H3` (src sha256 in `manifest.json`)
**Maturity:** REPRODUCED. **Class:** HARVEST_NOW.
**Status:** SPECIFICATION ONLY — nothing here is merged into any primary tree.

## 1. Integration surface (ASSUMED seam, UNVERIFIED)

- **Primary module:** ASSUMED — any thresholded propagation point on the
  primary where selected content crosses into consumer delivery or action.
  Candidate seams from the scratch copy (all UNVERIFIED against the live
  tree — Q3):
  - `core/cognitive_workspace.py` `Workspace.select(...)` — pressure-gated selection
  - `core/broadcast_integration.py` `CanonicalBroadcast.broadcast_concerns(...)` —
    the point where concerns cross into consumers
  - `core/self_model_port/ignition_arbitration.py` — name suggests an existing
    ignition concept on the primary; the primary lane must report what this
    module actually does before H3's gate is positioned relative to it.
- **Seam contract:** a function of `salience: float → {ignited: bool, strength: float}`.
  The gate sits BETWEEN selection and propagation: only `ignited=True`
  content propagates to consumers. The R1 path sits BELOW the gate:
  when nothing ignites, the tick may act on the graded arbitration winner
  (exploratory action) with ZERO consumer propagation — the gate's
  selectivity is untouched by the exploration path.

The lab's tick wires: `arbitrate → ignite(salience) → if ignited: broadcast;
else (R1): graded-winner action, no broadcast`. The R1 wiring is proven
post-K8 (WIRE decision 2026-10-07T07:23:05Z, all gates PASS on fresh seeds).

## 2. Interface contract

Vendored: `src/ignition.py` (`RecurrentIgnition`, `IgnitionRefused`).

```python
class RecurrentIgnition:
    def __init__(self, *, theta=0.6, gain=12.0, feedback=4.0,
                 max_cycles=50, eps=1e-6, linear_probe=False)
    def ignite(self, salience: float, jitter: float = 0.0) -> dict:
        # bistable recurrent loop: s_{n+1} = sigmoid(gain*(x_eff - theta) + feedback*(s_n - 0.5))
        # hard Heaviside: ignited = (s_final >= 0.5) — NO graded leak
        # returns {"salience","jitter","x_eff","strength","ignited","cycles","converged","mode"}
        # strength persists across ticks (reverberant seed = working-memory persistence)
    def reset_strength(self)
```

Parameters are FIXED at the lab-proven values (theta 0.45/0.6, gain 12,
feedback 4). **Do not tune theta upward expecting free wins** — the R2
adaptive-theta alternative was REJECTED (~2.7x ignitions, weakened
selectivity). Tuning theta is the known failure mode, not an improvement
knob.

R1 wiring contract (primary lane implements, lab `tick.py` is the reference):
```
winner, salience = arbitrate(stimuli)
rec = ignition.ignite(salience)
if rec["ignited"]:
    broadcast(item)              # full consumer path
else:
    action = act_on(winner)      # graded winner — action ONLY
    assert zero consumer deliveries on this trial   # the K8 gate (d) invariant
```

Adapter responsibilities:
- **Salience mapping:** the primary's selection margin / bid → `salience:
  float` in [0,1]-ish scale. The gate threshold theta=0.6 was proven against
  the lab's z-score bid scale; the primary must CALIBRATE its salience
  scale to the gate (see acceptance test 2) rather than retuning theta.
- **Jitter:** per-trial seeded jitter is a MEASUREMENT instrument for the
  K3 probability curve, not production behavior — production calls use
  `jitter=0.0`.
- **State ownership:** `strength` (reverberant) persists across ticks;
  serialize with the tick state. `reset_strength()` is the clean-state
  primitive (episode boundaries, restarts).
- **Error behavior:** non-finite salience → `IgnitionRefused`, fail closed
  (treat as quenched; receipt it).

## 3. Behavioral deltas + preregistered acceptance tests

Expected delta: a genuine ignition event (not weighted averaging) plus
freeze-free action selection without weakening the gate — NR-A-004's
"freeze is the price of the gate" is ruled out. Lab evidence: K3 Heaviside
step, transition width W 0.100 < 0.12 preregistered (5/5 fresh-seed repro
W 0.08–0.10); K8 R1 recovers frozen phases (phase-2 'c' fraction 0.0 →
0.79–0.81; totals 84–85 → 128–129) with zero non-feedback consumer
deliveries on 19–21 sub-ignition trials per seed.

Preregistration (stranger-runnable):
1. **Standalone reproduction:** `cp -r H3 /tmp/pkgtest && python3
   tests/test_h3.py` — must reproduce W=0.100 and R1 0.00→0.79/0.81/0.79
   with 0 violations.
2. **K3 probe on the primary's salience scale:** sweep salience with seeded
   jitter through the gate; measure propagation probability. Gate: Heaviside
   step with transition width W < 0.12 on the PRIMARY's scale. If the step
   is absent (graded curve), the gate is miscalibrated to the scale —
   calibrate the input, do not widen theta.
3. **Linear-probe kill control:** run the same sweep with
   `linear_probe=True` (graded passthrough). The step must VANISH under the
   probe. If it persists, the "step" was in the bid function, not the gate —
   REJECT the ignition claim.
4. **R1 sole-path invariant (gate d, ported):** on sub-ignition trials, assert
   zero deliveries to non-feedback consumers (the primary's analogue of the
   lab's 19–21 trials/seeds, 0 violations). The feedback consumer
   (attention_update / gain update) MAY receive sub-ignition content —
   learning continues; propagation does not.
5. **Non-degradation (gate c, ported):** the primary's existing selection
   benchmark (the K4 analogue) must not regress 4/4 seeds after wiring R1.

## 4. Rollback + tripwire

- **Rollback:** replace the gate with top-k pass-through (graded, no gate);
  revert the action selector to the hold-branch (pre-K8 behavior — the
  lab's post-wire reruns were byte-identical, so this is a clean revert).
- **Tripwire (proves harm):** monitor ignition rate per 100 ticks. If the
  rate exceeds 3x the preregistered baseline rate (the R2 signature:
  ~2.7x ignitions = weakened selectivity), or if sub-ignition consumer
  deliveries appear (gate-d violation), the gate is compromised — revert
  to hold-branch and investigate before re-enabling. Never "fix" a high
  ignition rate by raising theta: the R2 rejection is the evidence that
  this path degrades selectivity.

## 5. Bounds and risks (carried over verbatim, not softened)

- theta fixed at 0.45/0.6 (not ADAPTIVE — the R2 adaptive-theta
  alternative was REJECTED: ~2.7x ignitions, weakened selectivity).
  Do not tune theta upward expecting free wins.
- REPRODUCED 5/5 (K3); R1 confirmed on 3 fresh seeds 51501–51503
  (single lab) — no independent replication yet.
- The gate decides ALL propagation: any consumer path that bypasses it
  (side channel) invalidates the selectivity claim — the H2 K2 gate is
  the companion proof (see H2 spec).
