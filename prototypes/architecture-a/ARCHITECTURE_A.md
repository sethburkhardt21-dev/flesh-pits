# ARCHITECTURE A — WORKSPACE-CENTRIC PROTOTYPE

*Flesh Pits R&D lane, owner directive. Built 2026-10-07. Pure Python,
stdlib only, deterministic, seeded. Strict cognitive neutrality (§§14–15):
no identity-privileged machinery; verified by identity_symmetry_check.*

## What it is

```
perception -> specialists -> z-score salience bids -> attention arbitration
  (learned gains) -> BOUNDED buffer (K, eviction policy) -> recurrent
  ignition (bistable gate) -> BROADCAST (sole path) -> 6 consumers
  -> action (via planner queue, read through the bus) -> env -> reward
  -> feedback broadcast -> gain update (learning loop closed through broadcast)
```

## Files

| File | Role |
|---|---|
| `workspace_buffer.py` | True live bounded buffer (K or unbounded), explicit eviction policy (`lowest_bid_oldest_tiebreak` / `oldest`), newcomer competes unprotected, TTL decay, per-admission receipts |
| `bids.py` | RunningZScoreBid — change + habituation in one mechanism (sigmoid of causal running z-score), fail-closed on non-finite |
| `attention.py` | Arbitrator: habituated bids × learned gains, argmax, canonical tie-break; `update_gains()` delta rule (only observed channels, capped, frozen mode = fixed baseline) |
| `ignition.py` | Recurrent bistable loop → hard Heaviside gate on propagation; `linear_probe` kill-control |
| `broadcast.py` | In-memory sole-path bus: declared consumers, per-item consumer sets, lesion/restore, delivery receipts |
| `consumers.py` | 6 real consumers with instrumentable effects (memory store, self-state, planner queue, gain updater, consolidation marks, report) |
| `tick.py` | The tick; consumers exist only behind the bus; action selection reads the planner queue through the bus |
| `envs.py` | **PROVISIONAL** minimal env (canonical ENV_INTERFACE not on disk 2026-10-07); stationary-signal changing-relevance task |
| `experiments/k1..k4_*.py` | Kill experiments, §38 standard, receipts in `receipts/` |
| `experiments/identity_symmetry_check.py` | §§14–15 neutrality verification |

## Kill-experiment outcomes (all with numbers)

- **K1 capacity lesion — PASS.** P(target lost): K=3 → 0.000 / 0.507 / 1.000
  for distractor bids 0.50 / 0.85 / 0.95; K=inf → 0.000 in all conditions.
  Interference index 0 / +0.507 / +1.000, monotonic. The bound creates
  competitive interference; without it, none.
- **K2 broadcast lesion — PASS.** Selective lesion of memory_admit: that
  consumer 0 deliveries, all others ≈ baseline. Total lesion: all six
  consumers 0 at once, while buffer admissions (+90) and ignition (+90)
  continued. Recovery complete. No side channels — the sole-path claim holds.
- **K3 ignition probe — PASS.** Propagation probability is a Heaviside
  step: transition width 0.100 (< 0.12 preregistered), P=0 below,
  P=1 above. Graded control detected as linear (error 0.002) — the probe
  is not blind. There IS an ignition event, not weighted averaging.
- **K4 fixed-attention baseline — PASS 4/4 seeds.** Learned vs frozen
  total-reward ratios: 1.61, 2.03, 1.95, 1.57 (preregistered margin 1.30).
  Learned agent tracks the relevance reversal; frozen perseverates.

## Honest marks

- Learned attention's win is **bounded**: on signal-tracking reversals the
  frozen change-bid baseline adapts alone (NR-A-005). Learning earns its
  keep for salience-orthogonal relevance shifts — generalized 2026-10-07
  (K5) to noisy-signal tracking (P2, 4/4) and multi-reversal stationary
  shifts (P3, 4/4), but NOT to sparse delayed reward (NR-A-006: learned
  loses, R 0.08–0.17) or cue-conditioned contingencies without cue input
  (NR-A-007: no gap, R 1.03–1.09).
- **Stationarity perseveration** (NR-A-004) — RESOLVED 2026-10-07 (K6):
  R1 sub-ignition exploratory path recommended (act on the graded
  arbitration winner when nothing ignites; nothing unignited reaches
  consumers; gate still decides all propagation). 3/3 seeds: phase-2
  'c' fraction 0.0→0.78, total 85→128; K1–K3 reruns byte-identical PASS;
  K4 non-degradation 4/4. R2 (adaptive theta) also resolves but admits
  ~2.7× ignitions — rejected. The freeze is not the price of the gate.
- Three mechanism bugs found and fixed during the build (NR-A-001/002/003):
  tracking baselines, unobserved-arm punishment, within-tick ignition
  contamination. All in `research/negative_results.md`.
- `envs.py` is provisional; K4 must be re-run against the canonical
  ENV_INTERFACE when it lands.
- Maturity levels per component: see `MATURITY.md`. Nothing claims above
  its evidence. CONSCIOUSNESS: UNRESOLVED — this tests mechanisms.
