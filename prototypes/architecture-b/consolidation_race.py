"""EXP-FP-0005 — the consolidation race (FINAL_HANDOFF §9 item 8).

The audit (research/current_system_audit.md, memory section) gave
``consolidation_cycle (S-01 + memory_port variants)`` a KEEP_AND_DEEPEN
(race) disposition: "Two honest batch implementations; race them on one
corpus as the deepening experiment." This module runs it.

Two vendored implementations (race_vendored/, sha256-pinned in
PROVENANCE.md, byte-identical to the proven originals — NOT modified):

  P1 = S-01 batch cycle (malice_port): decay -> merge (relevance-desc
       greedy seed, cosine to seed) -> prune (summary layer only, input
       never mutated). Replay source: the promotion list (top-k summary
       ids by relevance), members expanded deterministically.
  P2 = memory_port cycle: decay -> merge (O(n^2) greedy, relevance sum)
       -> prune (in place). Replay source: get_replay_batch (top-k by
       relevance).

Arms (ONE fixed corpus per seed, shared by all arms):

  P1  prioritized replay via S-01 (promotion list)
  P2  prioritized replay via memory_port (top-k relevance batch)
  U   uniform replay (same replay budget K, seeded RNG)
  N   no offline replay

Preregistration (frozen BEFORE the run; sealed in the module, then in
the receipt):
  hypothesis: Offline prioritized consolidation replay causally improves
      later prediction performance on a held-out probe vs no replay, and
      beats uniform replay at equal budget.
  null: Offline replay changes nothing (no arm beats N); or prioritized
      replay does not beat uniform (priority is decoration).
  metric: IG_probe = probe_before - probe_after, where probe error =
      mean|eval_transition| on a FIXED 400-transition probe set drawn
      from random-policy pomaze rollouts (policy-independent, identical
      across arms and seeds), measured on a fresh ArchB whose weights saw
      exactly one online pass over the corpus before the offline phase.
      Higher IG = larger offline improvement.
  corpus: per seed s, 24 closed-loop pomaze episodes collected with a
      fixed reference ArchB (action_mode='active_inference',
      affect='none', seed=900+s). Identical corpus object for all arms.
  relevance (priority function, preregistered; per H5 risk note this is
      deployment-specific): after the online pass, pe_i =
      eval_transition(corpus transition i); max_pe = max pe_i (guard 0);
      rel_i = max(0.2, min(1.0, pe_i / max_pe)). Identical seeding for
      P1 and P2 (fair comparison of the consolidation mechanics).
  replay budget: K = 200 transitions per replay arm. P1: promoted
      summaries in promotion order, members ordered by (-seeded
      relevance, id), take first K. P2: get_replay_batch(k=200).
      U: uniform sample of 200 without replacement, RNG seeded
      (4242+s). N: none.
  seeds: [61701, 61702, 61703, 61704] — 4 fresh seeds, no overlap with
      battery (72701-72705), Phase-4 (301-303, 311-313, 321-323,
      331-333), or repro (51501-51503, 91511-91514, 91615-91618).
  decision rules:
    G1 (replay vs no replay): arm A beats N iff seed-mean IG_A >
        seed-mean IG_N AND A wins on >= 3/4 seeds (strict per-seed
        IG_A > IG_N). Tested separately for P1, P2.
    G2 (prioritization): same rule for P1-vs-U and P2-vs-U.
    G3 (implementation race): winner = argmax(seed-mean IG) over
        {P1, P2}; no-winner if |delta| < 1e-6.
    If G1 fails for BOTH P1 and P2: the Phase-4 battery claim
    ("consolidation CAUSAL") gets a BOUND — mechanism CAUSAL (per the
    brothel EXP-MEMORY-001 mechanism-level evidence), but no measured
    offline performance lift on this corpus. Recorded as a negative
    result, never reinterpreted.
  gates (run is VOID — not reinterpreted — if any fails):
    G0a: corpus length >= 300 transitions per seed.
    G0b: probe_before > 0 per seed (non-degenerate starting error).
    G0c: replay counts exactly K in P1/P2/U (recorded per seed).
    G0d: determinism spot-check — seed 61701 P1 recomputed; IG must
        match to 1e-12.
    G0e: fabrication-tripwire clean on consolidation_race.py +
        race_vendored/ before results are absorbed.
    G0f: hash-chained receipt written via the lab harness; verify_chain
        passes.

Usage:
    python3 consolidation_race.py

CONSCIOUSNESS: UNRESOLVED — mechanisms only.
"""

import copy
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "race_vendored"))
FP_EXP = os.path.join(HERE, "..", "..", "experiments")
sys.path.insert(0, FP_EXP)

from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, collect_transitions, _mean,
)
from agent import ArchB  # noqa: E402

import consolidation_cycle_s01 as s01  # noqa: E402
import consolidation_cycle_memory_port as mp  # noqa: E402

EXP_ID = "EXP-FP-0005"
SEEDS = [61701, 61702, 61703, 61704]
CORPUS_EPISODES = 24
PROBE_SIZE = 400
PROBE_SEED = 901
REPLAY_BUDGET = 200
REL_FLOOR = 0.2


def _collect_probe():
    rand_agent = make_agent(ENVS_BY_NAME["pomaze"], seed=42,
                            action_mode="random", affect="none")
    return collect_transitions("pomaze", 6, PROBE_SEED,
                               rand_agent)[:PROBE_SIZE]


def _probe_error(agent, probe):
    return _mean([agent.eval_transition(o, a, o2)
                  for (o, a, o2, r) in probe])


def _build_corpus(seed):
    ref = make_agent(ENVS_BY_NAME["pomaze"], seed=900 + seed,
                     action_mode="active_inference", affect="none")
    return collect_transitions("pomaze", CORPUS_EPISODES, seed, ref)


def _fresh_agent(seed):
    cls = ENVS_BY_NAME["pomaze"]
    space = cls().observation_space()
    n_actions = cls().action_space()["n"]
    return ArchB(observation_space=space, n_actions=n_actions,
                 env_name="pomaze", seed=7000 + seed, affect="none")


def _seed_relevances(agent, corpus):
    pes = [agent.eval_transition(o, a, o2) for (o, a, o2, r) in corpus]
    m = max(pes) if pes else 0.0
    if m <= 0.0:
        return [REL_FLOOR] * len(corpus)
    return [max(REL_FLOOR, min(1.0, pe / m)) for pe in pes]


def _arm_p1_s01(corpus, rels, budget):
    """Prioritized replay via S-01: promotion list -> members."""
    entries = [{"id": f"e{i}", "vector": list(o),
                "relevance": float(rels[i]), "payload": i}
               for i, (o, a, o2, r) in enumerate(corpus)]
    rel_by_id = {f"e{i}": float(rels[i]) for i in range(len(corpus))}
    cfg = s01.CycleConfig(decay_rate=0.99, merge_threshold=0.9,
                          prune_threshold=0.1, max_summaries=10000,
                          promotion_k=10000)
    cyc = s01.ConsolidationCycle(cfg)
    res = cyc.run(entries, tick=0)
    by_sid = {s["summary_id"]: s for s in res.summaries}
    chosen, seen = [], set()
    for sid in res.promotion:
        s = by_sid[sid]
        members = sorted(s["member_ids"],
                         key=lambda mid: (-rel_by_id[mid], mid))
        for mid in members:
            idx = int(mid[1:])
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
            if len(chosen) >= budget:
                break
        if len(chosen) >= budget:
            break
    return chosen, {"summaries": len(res.summaries),
                    "overflow": len(res.overflow),
                    "promotion": len(res.promotion),
                    "cycles_run": cyc.cycles_run}


def _arm_p2_memory_port(corpus, rels, budget):
    """Prioritized replay via memory_port: consolidate + top-k."""
    entries = [{"id": f"e{i}", "vector": list(o),
                "relevance": float(rels[i]), "payload": i}
               for i, (o, a, o2, r) in enumerate(corpus)]
    cyc = mp.ConsolidationCycle(decay_rate=0.99, merge_threshold=0.9,
                                prune_threshold=0.1)
    kept = cyc.consolidate(copy.deepcopy(entries))
    batch = cyc.get_replay_batch(kept, k=budget)
    chosen = [e["payload"] for e in batch]
    return chosen, {"consolidated": len(kept),
                    "batch": len(batch),
                    "cycles": cyc.consolidation_count}


def _arm_u_uniform(corpus, budget, seed):
    rng = random.Random(4242 + seed)
    return rng.sample(range(len(corpus)), budget), {"draws": budget}


def _run_arm(corpus, chosen, probe, seed):
    agent = _fresh_agent(seed)
    for (o, a, o2, r) in corpus:
        agent.learn_transition(o, a, o2, r)
    before = _probe_error(agent, probe)
    rels = _seed_relevances(agent, corpus)
    for idx in chosen:
        o, a, o2, r = corpus[idx]
        agent.learn_transition(o, a, o2, r)
    after = _probe_error(agent, probe)
    return before, after, rels


def run_seed(seed, probe):
    corpus = _build_corpus(seed)
    if len(corpus) < 300:
        return {"seed": seed, "gate_fail": "G0a: corpus too small",
                "corpus_len": len(corpus)}
    # P1 and P2 share relevance seeding procedure; compute once via a
    # throwaway identically-initialized agent for the chosen lists.
    probe_agent = _fresh_agent(seed)
    for (o, a, o2, r) in corpus:
        probe_agent.learn_transition(o, a, o2, r)
    before = _probe_error(probe_agent, probe)
    if before <= 0.0:
        return {"seed": seed, "gate_fail": "G0b: probe_before == 0",
                "corpus_len": len(corpus), "probe_before": before}
    rels = _seed_relevances(probe_agent, corpus)
    chosen_p1, diag_p1 = _arm_p1_s01(corpus, rels, REPLAY_BUDGET)
    chosen_p2, diag_p2 = _arm_p2_memory_port(corpus, rels, REPLAY_BUDGET)
    chosen_u, diag_u = _arm_u_uniform(corpus, REPLAY_BUDGET, seed)
    arms = {"P1": chosen_p1, "P2": chosen_p2, "U": chosen_u, "N": []}
    for arm, chosen in arms.items():
        if arm != "N" and len(chosen) != REPLAY_BUDGET:
            return {"seed": seed,
                    "gate_fail": f"G0c: arm {arm} replayed "
                                 f"{len(chosen)} != {REPLAY_BUDGET}",
                    "corpus_len": len(corpus)}
    out = {"seed": seed, "corpus_len": len(corpus),
           "probe_before": before, "diags": {"P1": diag_p1,
                                             "P2": diag_p2, "U": diag_u},
           "arms": {}}
    for arm, chosen in arms.items():
        b, a, _ = _run_arm(corpus, chosen, probe, seed)
        assert abs(b - before) < 1e-12, \
            f"probe_before drift across arms: {b} vs {before}"
        out["arms"][arm] = {"probe_after": a, "IG": b - a,
                            "replayed": len(chosen)}
    return out


def main():
    probe = _collect_probe()
    assert len(probe) == PROBE_SIZE, f"probe size {len(probe)}"
    results = [run_seed(s, probe) for s in SEEDS]
    failed = [r for r in results if "gate_fail" in r]
    # G0d: determinism spot-check — recompute seed 61701 P1.
    corpus = _build_corpus(SEEDS[0])
    probe_agent = _fresh_agent(SEEDS[0])
    for (o, a, o2, r) in corpus:
        probe_agent.learn_transition(o, a, o2, r)
    rels = _seed_relevances(probe_agent, corpus)
    chosen1, _ = _arm_p1_s01(corpus, rels, REPLAY_BUDGET)
    chosen2, _ = _arm_p1_s01(corpus, rels, REPLAY_BUDGET)
    ig_a = _run_arm(corpus, chosen1, probe, SEEDS[0])
    ig_b = _run_arm(corpus, chosen2, probe, SEEDS[0])
    det_ok = abs((ig_a[0] - ig_a[1]) - (ig_b[0] - ig_b[1])) < 1e-12
    print(json_summary(results, failed, det_ok))
    return results, failed, det_ok


def json_summary(results, failed, det_ok):
    import json
    rows = []
    for r in results:
        if "gate_fail" in r:
            rows.append({"seed": r["seed"], "GATE_FAIL": r["gate_fail"]})
            continue
        rows.append({"seed": r["seed"], "corpus_len": r["corpus_len"],
                     "probe_before": round(r["probe_before"], 6),
                     **{f"IG_{a}": round(r["arms"][a]["IG"], 6)
                        for a in ("P1", "P2", "U", "N")}})
    return json.dumps({"gate_failures": [f["gate_fail"] for f in failed],
                       "determinism_spot_check_ok": det_ok,
                       "rows": rows}, indent=1)


if __name__ == "__main__":
    main()
