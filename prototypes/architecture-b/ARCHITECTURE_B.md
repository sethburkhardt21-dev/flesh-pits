# ARCHITECTURE B — Predictive-Model-Centric (Brothel prototype v1)

*Flesh Pits R&D lane, owner directive §§23, 30, 33, 35, 38. Implemented 2026-10-07.*

## Data flow

```
obs_t --> context key (categorical values)
mu0 (L0 state belief, persists) + action --> L0 prediction W.f
                                            + L1 top-down D_ctx.f
                                            --> xhat (next-obs prediction)
                                            --> rhat = w_r.f + b_r + R_ctx[a]
obs_{t+1} --> e0 = obs - xhat            (perception IS prediction error)
              e1 = e0 - D_ctx.f          (residual L1 failed to explain)
              rerr = r - rhat
pi0/pi1/piR estimated from recent error statistics (never hand-set)
dW     = eta0 * (pi0*e0) (x) f
dD_ctx = etaD * (pi1*e1) (x) f        (active context only)
dw_r   = eta_r * piR * rerr * f
dR     = etaR * piR * rerr            (taken action only)
mu0   <- xhat + e0 / (1 + pi0)        (Kalman-like posterior, bounded)
memory: encode (obs, a, r, e0, e1, pi, attention=surprise) w/ provenance
        retrieve-k-similar (cosine on obs) -> correct xhat, predict per-action error
action: argmin_a G(a) = -Vhat(a) - lamIG*IGhat(a) + lamR*Risk
        IGhat from episodic per-action error history (information gain)
affect: valence/arousal from error dynamics -> explore/learn/persist knobs
        (PAD dictionary kept as the competing baseline controller)
```

## Components

| File | Component | Status |
|---|---|---|
| `generative_model.py` | HierarchicalGenerativeModel (L0/L1, persistent mu0, context experts) | built, tested |
| `precision.py` | PrecisionEstimator (pi from error stats; uniform ablation hook) | built, tested |
| `memory.py` | EpisodicStore (port of experience_store algorithm + provenance) | built, tested |
| `predictions.py` | §30 targets: next-obs, action-consequence, retrieval-usefulness; per-tick JSONL | built, tested |
| `active_inference.py` | ActiveInferenceSelector (EFE proxy; greedy/random modes for the kill test) | built, tested |
| `affect.py` | ErrorAffect (error-derived) + PadController (donor-exact baseline) | built, tested |
| `agent.py` | ArchB (Agent ABC: act/update/snapshot/restore) | built, tested |

## Design decisions (and reversals)

1. **v1 MoE L1 → v2 context experts (NR-B-001).** The first L1 was a
   mixture-of-experts with input-free gating (mu1 vector). It had a fatal
   cold-start: the gating weights and the experts needed each other for
   gradient, so cue×action contingencies never learned (reward predictions
   flat across actions after 200 trials). Replaced with context-indexed
   experts: the active context's expert updates directly — clean credit
   assignment, no chicken-and-egg.
2. **mu0 posterior as Kalman blend (not kappa·pi·e).** The first form
   (`mu0 <- xhat + kappa*pi*e0`) detonated numerically (pi up to 100 ×
   e0 → overflow in 3 ticks). Replaced with the bounded convex blend
   `mu0 <- xhat + e0/(1+pi0)`.
3. **Learning-rate rescaling.** eta0 0.05→0.005, etaD 0.02→0.002, etc.:
   stability requires eta·pi·||f||² < ~2; the original rates with
   pi up to 100 diverged (reward weights overflowed at tick 69).
4. **Affect isolated from world-model tests (NR-B-002).** The error-derived
   affect's arousal saturates at 1.0 on volatile tasks → explore_gain=2.0
   permanently → the agent explores forever and sits at chance (20.0 vs
   26.0 with affect off on changing_rule). World-model kill experiments
   run with affect="none"; the affect controller is judged only by K4.
5. **Context = all categorical fields.** For changing_rule this gives
   (cue, last_action) = 6 contexts (fragmented but functional). Documented
   limitation; first-categorical-only is the candidate refinement.

## Load-bearing claims → experiments

1. Error declines with experience → EXP-AB-C1 (delayed_reward, grid_world).
2. Precision beats uniform → EXP-AB-C2 (changing_rule, 10% noise).
3. Hierarchy matters → EXP-AB-K3 + EXP-AB-BAR (L1 lesion).
4. Active inference beats random/greedy on IG → EXP-AB-C4 (pomaze).
K1 (freeze), K2 (learned vs fixed), K5 (shuffle), K4 (affect vs PAD).

## Honest limitations

- Single seeds throughout (compute budget); reproduction across seeds is
  the next step before any harvest claim.
- The §30 per-tick log for the BAR run is in
  `experiments_out/EXP-AB-BAR.predictions.jsonl` (3 targets × ticks).
- CONSCIOUSNESS: UNRESOLVED — mechanisms only, no narratives.
