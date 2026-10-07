# ARCHITECTURE BAKE-OFF — FINDINGS (Phase 4 verdict, 2026-10-07)

*Directive §§23–25, 40–41. Coordinator synthesis from five worker lanes, all receipts on disk. Every kill below was preregistered; every number measured. CONSCIOUSNESS: UNRESOLVED throughout — mechanisms only.*

## Verdict

| Architecture | Fate | Basis |
|---|---|---|
| **A — workspace-centric** | **STRONGEST SURVIVOR** | K1–K4 all PASS; K4 re-run PASS 4/4 on canonical env. Maturity: buffer/broadcast/ignition/consumers CAUSAL, attention GENERALIZING (K5: noisy-signal + multi-reversal generalize 4/4; delayed_reward loses structurally NR-A-006; changing_rule no-gap without cue input NR-A-007), bids CAUSAL for arbitration dynamics (K7: winner entropy → 0.0 bits — but learned gains compensate for dead bids; the K4 win is carried by the gain loop NR-A-009). NR-A-004 RESOLVED: R1 sub-ignition exploratory path recommended (freeze recovery 0.0→0.78, K1–K3 byte-identical, K2 sole-path untouched) — not yet wired into default tick, pending decision. |
| **B — predictive-model-centric** | **MIXED — core survives, ornaments killed** | K1, K2, K5 survive; C1 holds. K4 (error-affect), C2 (precision), C4 (active-inference label) KILLED by its own experiments. K3 weak at short horizon but hierarchy EARNS ITS KEEP on longer horizons (K3B: reward-channel lesion gap +0.0602, 3/3 seeds, ~19× K3). Memory proven CAUSAL by ablation (M1: selective deficit +2.200 pomaze, 3/3 seeds). |
| **D — lean null** | **The bar, honestly weak** | Loses to symbolic everywhere; two falsifiable weaknesses isolated and preserved. |
| **C — hybrid** | **DEFERRED by design** | Merger of proven parts only — premature to build before A/B verdicts. Candidate components: A's workspace machinery + B's learned predictor. |

## What the evidence says, per contender

**A.** The workspace bound does causal work (K1: interference 0→0.507→1.0 at K=3, exactly 0 at K=∞). Broadcast-as-sole-path holds in the prototype (K2: total lesion silences all six consumers at once; I re-ran this myself — PASS, numbers match). Ignition is a real threshold event (K3: Heaviside step, width 0.100 < 0.12 preregistered). Learned attention beats fixed 1.57–2.03× on salience-orthogonal relevance shifts (K4; canonical re-run 1.44–1.70, all ≥1.30). Bounds: K4's claim is specific to salience-orthogonal shifts (on signal-tracking reversals the frozen baseline adapts alone — NR-A-005); theta=0.6 stationarity perseveration is an open design question (NR-A-004); single task family, not generalized.

**B.** Learning is real (K1: 26.2 > 21.6 frozen — not a static function). Learned beats fixed predictor (K2: |e| 0.98 < 1.32). Error declines with learning (C1 holds, 2 envs). Hierarchy beats chance on changing-rule (BAR: 24.4 > 21.4 > 20.0) but the effect is WEAK (K3: +0.0032, metric diluted). Killed: precision weighting is HARMFUL under distribution shift (C2: uniform 32.1 > precision 26.3 — replicated by the coordinator's own hand at 26.300 vs 32.100; overshoot after rule flips); error-derived affect LOSES to the PAD dictionary as a controller (K4: −0.762 > −0.787 — the elegant story loses to the cheap baseline); the "active inference" label comes off (C4: greedy IG 0.0502 > AI 0.0275). Honest mid-build redesign preserved (NR-B-001: mixture-of-experts cold-start → context experts). Single seeds throughout — nothing reproduced.

**D.** 187 lines. Predictor → error → episodic store → K=4 top-k → greedy. Its value is as a characterized floor: linear predictor cannot learn cue×action (chance 20/40 on changing-rule); pure-greedy locks into no-op fixed points (0.0000 on delayed-reward). B partially cleared these bars; that is what "beating D" concretely meant.

## Cross-cutting findings (lab law now)

1. **Broadcast-consumer rule:** a workspace/broadcast claim without measured downstream consumer change is theater. Confirmed twice — current pipeline Δ=0 bit-identical; A's prototype passes K2.
2. **Precision-from-error-statistics is harmful under distribution shift** in the tested implementation. The literature's "most portable mechanism" needs a shift-robustness caveat. Genuine finding, not a bug.
3. **PAD dictionary beats error-derived affect.** Keep PAD; drop the story.
4. **The 0.7 attention gate is bookkeeping-only** — low-attention memories remain retrievable. Labeling correction.
5. **Memory write path:** provenance CAUSAL, consolidation CAUSAL, forget CAUSAL; tag-at-encoding via replay/PE and PE-gated retrieval are UNREALIZED (no code) — build targets. The 'agent' provenance origin is unreachable via the public API — needs architect attention.
6. **"WIRED (enrichment only)" ≠ causally wired.** Live-root verification: wave-3b modules execute in the tick but outputs are explicitly enrichment-only — *"reported, never applied."* Enrichment is telemetry, not wiring. The lesion-or-drop rule stands.

## What would change this verdict

- Multi-seed reproduction of every Phase-4 kill (nothing is REPRODUCED yet — the single largest evidence gap).
- B's hierarchy under longer horizons / harder tasks; sharper C2/C4 metrics.
- A's learned attention on delayed-reward, noisy, and multi-reversal tasks.
- The 1.5B model-independence test (0.5B↔1.5B transfer) — rung not yet downloaded.
- C built from the surviving parts, beaten against A and B by preregistered margins.
