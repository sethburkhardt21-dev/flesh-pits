# ARCHITECTURE A — CAUSAL MATURITY LEDGER

*Directive §41. Updated 2026-10-07 after K1–K4. Levels: PRESENT →
EXECUTED → INTEGRATED → CAUSAL → ADAPTIVE → GENERALIZING → REPRODUCED.
Marked honestly; nothing above its evidence.*

| Component | Level | Evidence |
|---|---|---|
| `workspace_buffer.py` (bounded buffer, eviction policy) | **CAUSAL** | K1: interference 0→0.507→1.0 monotonic in distractor bid at K=3, exactly 0 at K=inf. Lesioning capacity causally changes competition. Not ADAPTIVE (K, policy fixed). |
| `bids.py` (RunningZScoreBid: change + habituation) | **INTEGRATED** | Executed every tick; output consumed by arbitration; habituation measured (constant stimulus → 0.5). CAUSAL pending an explicit bid-ablation (lesion bids → arbitration must change). |
| `attention.py` (arbitration + learned gains) | **ADAPTIVE** | K4: freezing gains changes total reward 1.57–2.03× (CAUSAL). Gains move with utility history via broadcast feedback (gain_history trail = ADAPTIVE). Not GENERALIZING (one task family). |
| `ignition.py` (recurrent bistable gate) | **CAUSAL** | K3: hard Heaviside step, W=0.100; gate causally determines propagation. Not ADAPTIVE (theta/gain/feedback fixed). |
| `broadcast.py` (sole-path bus) | **CAUSAL** | K2: lesion_all → all six consumers silent at once; internal processing continues. No side channels detected. |
| consumers: memory_admit, self_model_update, planner_input, consolidation_eligible, report | **CAUSAL** | K2: each consumer's instrumented effect appears/vanishes exactly with its broadcast deliveries. |
| consumer: attention_update | **ADAPTIVE** | Closes the K4 learning loop; gains move only through its broadcast feedback. |
| `tick.py` (sole-path wiring, closed loop) | **INTEGRATED** | Full pipeline executes closed-loop; broadcast verified as sole content path (K2). Whole-tick CAUSAL not separately claimed. |
| `envs.py` (ChangingRelevanceEnv) | **EXECUTED** | Provisional local env (canonical ENV_INTERFACE not on disk as of 2026-10-07). K4 must be RE-RUN against the canonical spec when it lands. |
| identity symmetry (§§14–15) | **CAUSAL** (test) | Bit-identical trajectories for novel labels incl. a historically privileged string; tie-break is label-agnostic channel order. |

## What failed or is BLOCKED

- **Bid-ablation for CAUSAL bids**: not yet run (bids stay INTEGRATED).
- **Generalization of learned attention**: only the stationary-signal
  reversal task; untested on delayed-reward, noisy, or multi-reversal
  tasks. K4's claim is bounded: learning is load-bearing for
  *salience-orthogonal* relevance shifts (NR-A-005).
- **theta=0.6 stationarity perseveration**: architectural property, not
  a bug — ignition-gated action selection cannot act under full
  stationarity. Open design question for Phase 4.
- **Canonical environment**: BLOCKED on the environments worker's
  ENV_INTERFACE.md (not on disk). K4 re-run pending.
- Kuramoto/komplex binding, coherence-gated attention: NOT built —
  per the candidates doc they are EXPERIMENT_AGAINST, and nothing in
  K1–K4 needed them. They stay out unless a lesion demands them.
