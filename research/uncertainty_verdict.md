# Uncertainty verdict — Track D (metacognition hazard program)

**Date:** 2026-10-07. **Lane:** uncertainty-hazard. **Status:** COMPLETE.
**Question:** which estimator families produce *decision-usable* uncertainty —
probabilities that beat sequential climatology AND change a downstream
decision for the better?

## Verdict: none. No family produces decision-usable uncertainty on this protocol.

The honest baseline stands: B's §30 stated uncertainties are miscalibrated
(EXP-FP-CALIB-01), and nothing tested here displaces the running base rate
as the decision-usable probability.

## Scoreboard (preregistered gate: Brier < sequential climatology on ≥3/4 of
the four CALIB-01 domains; 1950 ticks/domain; corr(sigma,|err|) reported)

| ID | Family | Gate | corr(sigma,\|err\|) | Decision-use | Maturity |
|---|---|---|---|---|---|
| EXP-FP-0110 | (a) conformal transducer (rolling fit/calibration) | REJECT 0/4 | 0.02–0.07 | — (gate failed) | REJECTED |
| EXP-FP-0111 | (b) rolling-window empirical rate (W=200) | REJECT 2/4 | 0.02–0.10 | — (gate failed) | REJECTED |
| EXP-FP-0112 | (c) ensemble vote fraction (as-run) | **VOID** — contemporaneous-outcome leakage (p_t used members' realized errors at tick t) | — | — | VOID (instrument invalid; numbers not interpreted) |
| EXP-FP-0113 | (d) phase-stratified climatology (episode mod 5) | **PASS 3/4** (thin: +0.003/+0.005/+0.019) | 0.02–0.10 | DU1 NO-WIN (fires <1.5%, hurts when it fires); DU2 NO-WIN (apply rate 0.000, cold-start collapse) | INTEGRATED at best — **not decision-useful** |
| EXP-FP-0114 | (c) honest shadow-mode ensemble (pre-outcome disagreement → Gaussian bridge) | REJECT 0/4 (D3 UNMEASURABLE) | **0.29 on D1** (strongest coupling in the program) | — (gate failed) | REJECTED |

## Exact bounds

1. **Conformal prediction** (one-sided transducer, rolling F=100/C=200;
   clean-room stdlib reimplementation — MAPIE/D17 is ADOPT but NOT vendored,
   H11 is PACKAGED ONLY): adds nothing over the running average on a
   near-stationary stream (0/4; skills −0.005…−0.272). Well-calibrated on
   synthetic iid data yet still loses to climatology there — the failure is
   structural (estimation noise > adaptivity gain), not a bug.
   Bound: distribution-free ≠ better-than-climatology.
2. **Rolling empirical rate** (W=200): 2/4, both marginal (+0.002/+0.003).
   The W=50 ablation reaches CALIBRATED on D4 alone (the program's only
   CALIBRATED verdict outside the voided run). Bound: recency is not signal
   on changing_rule — event rates are too stable for windowing to help.
3. **Ensemble disagreement** (N=5, init seeds): as-run VOID (leakage —
   sibling outcomes correlate, so vote fractions peeked at the answer; the
   4/4 "PASS" was artifact). Honest redesign (shadow-mode, pre-outcome):
   members converge to near-identical predictions (disagreement ~25× too
   small for the Gaussian bridge; p saturates; D3 degenerate). The residual
   disagreement couples to error at corr 0.29 (D1) — ranking signal with
   catastrophically wrong scale. Bound: init-seed diversity washes out under
   shared-trajectory learning; disagreement needs scale calibration to be
   usable, which is a future experiment, not a retrofit.
4. **Phase-stratified climatology** (the lane's own idea): the only gate
   PASS (3/4), via the flip-cycle structure (mod-2 ablation: 0/4). But
   decision-inert: DU1 deferral fires on <1.5% of ticks and hurts (−1.6/−4.1
   where it fires); DU2 retrieval gating collapses to never-apply
   (cold-start feedback: p_init=0.5 ≯ 0.5 → block → benefit 0 → p→0 —
   the EXP-FP-0006 degeneracy recurring with an external probability).
   Bound: beats climatology thinly, changes no decision → INTEGRATED at
   best, NOT decision-useful.

## Methodological catches (first-class results)

- **0112 leakage:** a preregistered design can still be invalid. The
  contemporaneous-outcome leak was caught at interpretation, the run VOIDed
  before absorption, and the redesign preregistered separately (0114).
  Rule: when an uncertainty estimator looks dramatically better than
  climatology on the first try, audit what p_t is allowed to see.
- **DU2 cold-start collapse:** gating on a live-trained p with a strict
  threshold and 0.5-init never bootstraps. The lab already adjudicated the
  fix class (shadow estimator, EXP-FP-0006→0007); no retrofit was smuggled
  into this program.

## The one positive lead

0114's corr(sigma_ens,|err|)=0.29 with ~25× scale error: ensemble
disagreement has ranking signal at the wrong scale. A scale-calibrated
disagreement mapping (learned scale factor or rank-based probability),
preregistered as its own experiment with the same gate + DU protocol, is
the next resolving experiment. Everything else in this program is closed.

## Provenance

- Preregistrations: `experiments/preregistration_EXP-FP-011{0,1,2,3,4}.json`
  (0110–0113 sealed pre-run; 0114 sealed post-void, pre-rerun).
- Driver: `prototypes/architecture-b/exp_uncertainty_trackd.py`
  (stdlib only; fabrication-tripwire CLEAN; G1 determinism PASS).
- Gate scoring: `benchmarks/calibration_battery.py::score_domain` (verbatim reuse).
- Receipts: `receipts/EXP-FP-011{0,1,2,3,4}.json` (hash-chained; own-hash and
  prev-link verified; historical chain issues preserved untouched per lab law).
- Registry: `experiments/EXPERIMENT_REGISTRY.md` (Track D section + index rows).
- Negative results: `research/negative_results.md` (5 entries).
- IDs: `EXP-FP-011x` family claimed via `experiments/id_registry.py`
  (lane `uncertainty-hazard`); no collisions.
- DU protocols: DU1 (defer on p_D3>0.5), DU2 (retrieval gate on p_D4>0.5),
  τ=0.5 frozen; 4 fresh seeds {73701–73704}; win = seed-mean Δ>0 and ≥3/4 agree.
- Maturity scale: PRESENT → EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE →
  GENERALIZING → REPRODUCED. Nothing in this program advances past
  INTEGRATED (0113); the rest are REJECTED or VOID.
