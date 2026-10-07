"""K8 — R1 WIRING CONFIRMATION (NR-A-004 decision experiment).

PREREGISTRATION (written before the run; seeds fresh, never used before).

Context: K6 (seeds 1111/2222/3333) recommended wiring R1 — the
sub-ignition exploratory path — into Architecture A's default tick:
when nothing ignites, act on the graded arbitration winner WITHOUT
propagating anything to consumers (the gate still decides ALL
propagation; K2 sole-path unaffected). R2 (adaptive theta) was
REJECTED (admits ~2.7x ignitions, weakens gate selectivity). This
experiment confirms R1 on FRESH seeds and decides: WIRE or REJECT.

Hypothesis: R1 preserves gate selectivity, the K2 sole-path
guarantee, and K1-K4 verdicts, while recovering frozen phases at
theta=0.6.

Null: R1 either (i) fails to recover the freeze on fresh seeds, or
(ii) alters the ignition gate's selectivity, or (iii) leaks workspace
content to consumers on sub-ignition trials, or (iv) degrades the K4
learned-attention win below R 1.30. ANY of these -> REJECT R1.

Candidate: SubIgnitionExploreTick imported from k6 (the exact class
K6 tested — no re-implementation drift). The default WorkspaceTick is
NOT modified by this experiment; wiring happens only after a WIRE
decision.

A NOTE ON GATE (a) — selectivity, sharpened honestly: the task asks
for "ignition counts identical with/without R1." Under LEARNED gains
that gate would be dishonest: R1's recovery works precisely BY
feeding the reward->feedback->gain loop, so the bid trajectory (and
hence ignition counts) MUST diverge for recovery to happen —
demanding identity under learning demands no recovery. The
mechanism-level selectivity claim is: identical bids => identical
ignition decisions. Frozen gains isolate exactly that (bids become a
pure function of observations; the R1 path can change actions but not
bids). So (a1) preregisters EXACT ignition-trajectory identity under
frozen gains, and (a2) records learned-gains ignition counts as
direction evidence (R1 >= baseline expected via recovery), not as an
identity gate.

Metrics (all on fresh seeds {51501, 51502, 51503}, 200 ticks,
theta=0.6, stationary-signal changing-relevance, K4 config):

  (a1) FROZEN-gains ignition identity: baseline WorkspaceTick vs R1
       candidate, frozen_gains=True. Per-tick ignited-flag sequences
       must be EXACTLY identical (byte-identical) on 3/3 seeds.
       PASS requires 3/3.
  (a2) Learned-gains ignition counts: recorded for baseline vs R1;
       expected direction R1 >= baseline (recovery moves gains, gate
       sees more threshold crossings). Not gated as identity.
  (b) K1-K3 reruns byte-identical: run the ORIGINAL k1/k2/k3 scripts
      (default tick still unwired) and byte-compare the receipts to
      the on-disk receipts. PASS requires byte-identical K1, K2, K3.
      (Establishes the pre-wire baseline; post-wire K2 is handled
      honestly in the wire step if its closed-loop dynamics shift.)
  (c) K4 non-degradation with the R1 candidate at theta=0.45
      (K6's k4_with_tick): R >= 1.30 on >= 3/4 seeds {11,22,33,44}.
  (d) Sole-path on sub-ignition trials: on EVERY tick whose
      action_selection.path == "sub_ignition_explore", the bus
      delivery log must contain zero deliveries with kind !=
      "feedback" and zero deliveries to any consumer other than
      attention_update. (The per-tick feedback item to
      attention_update is pre-existing machinery, present in the
      baseline on every tick; it carries reward utility through the
      designed learning channel, not workspace content.) Also assert
      the planner queue was empty on those ticks (definitional).
      PASS requires 100% of sub-ignition trials on all seeds.
  (e) Freeze-recovery replication: phase-2 'c' fraction
      (ticks 101..200 acting on 'c'): baseline ~0.0 -> R1 > 0.5 on
      3/3 seeds; total reward R1 > baseline on 3/3 seeds.

Decision rule: WIRE iff (a1) AND (b) AND (c) AND (d) AND (e) all
pass. Otherwise REJECT with the failing gate(s) stated.

Kill-style: a REJECT is recorded in research/negative_results.md as a
new NR-A entry (expected/happened/rules-out/lesson).
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists
from workspace_buffer import BoundedWorkspace
import itertools
from k6_stationarity_resolution import (
    SubIgnitionExploreTick, k4_with_tick)


def reset_item_ids():
    """Reset the class-level item-id counter so two independent
    conditions in one process get comparable ids. Test-harness only;
    changes no behavior (ids are unique labels, never content)."""
    BoundedWorkspace._ids = itertools.count(1)

SEEDS = (51501, 51502, 51503)
TICKS = 200
THETA = 0.6
K4_SEEDS = (11, 22, 33, 44)
K4_MARGIN = 1.30
K4_NEED = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run_frozen_identity(seed):
    """(a1): frozen gains — ignition trajectory must be identical.

    Compares behavioral content per tick: which bids ignited and the
    arbitration winner (not raw item_ids — those ride a process-global
    counter, reset here so they are comparable too)."""
    channels = ["a", "b", "c"]
    seqs = {}
    for name, cls in (("baseline", WorkspaceTick),
                      ("R1", SubIgnitionExploreTick)):
        reset_item_ids()
        wk = cls(channels, make_specialists(channels), capacity=3,
                 frozen_gains=True, gain_lr=0.15,
                 ignition_kwargs={"theta": THETA})
        env = ChangingRelevanceEnv(channels, TICKS, seed=seed,
                                   stationary_signals=True)
        sig = []
        for _ in range(TICKS):
            tr = wk.step(env.observe(), env)
            ign = tuple(sorted(
                (i["item_id"], round(i["strength"], 9))
                for i in tr["ignitions"] if i["ignited"]))
            bids = tuple(sorted(
                (k, round(v, 9))
                for k, v in tr["arbitration"]["competed_bids"].items()))
            sig.append((ign, bids, tr["arbitration"]["winner"]))
        seqs[name] = sig
    identical = seqs["baseline"] == seqs["R1"]
    n_base = sum(len(s[0]) for s in seqs["baseline"])
    n_r1 = sum(len(s[0]) for s in seqs["R1"])
    return identical, n_base, n_r1


def run_learned(seed):
    """(a2)/(d)/(e): learned gains — recovery + sole-path instrumentation."""
    channels = ["a", "b", "c"]
    out = {}
    for name, cls in (("baseline", WorkspaceTick),
                      ("R1", SubIgnitionExploreTick)):
        wk = cls(channels, make_specialists(channels), capacity=3,
                 frozen_gains=False, gain_lr=0.15,
                 ignition_kwargs={"theta": THETA})
        env = ChangingRelevanceEnv(channels, TICKS, seed=seed,
                                   stationary_signals=True)
        sub_ign_ticks = []
        for _ in range(TICKS):
            tr = wk.step(env.observe(), env)
            if tr.get("action_selection", {}).get("path") \
                    == "sub_ignition_explore":
                sub_ign_ticks.append(tr["tick"])
        # (d): sole-path audit on sub-ignition trials
        violations = []
        for t in sub_ign_ticks:
            for d in wk.bus.deliveries:
                if d["tick"] != t or d["status"] != "delivered":
                    continue
                if d["kind"] != "feedback" \
                        or d["consumer"] != "attention_update":
                    violations.append(
                        {"tick": t, "kind": d["kind"],
                         "consumer": d["consumer"]})
        acts = wk.actions
        p2 = acts[TICKS // 2:]
        out[name] = {
            "ignited": wk.total_ignited,
            "sub_ignition_trials": len(sub_ign_ticks),
            "sole_path_violations": violations,
            "phase2_c_fraction": round(
                sum(1 for a in p2 if a == "c") / len(p2), 4),
            "total_reward": round(sum(wk.rewards), 2),
        }
    return out


def rerun_k1_k2_k3():
    """(b): rerun original scripts, byte-compare receipts."""
    receipts = {
        "k1": "receipts/k1_capacity_lesion.json",
        "k2": "receipts/k2_broadcast_lesion.json",
        "k3": "receipts/k3_ignition_probe.json",
    }
    before = {k: sha256(os.path.join(TREE, v))
              for k, v in receipts.items()}
    scripts = {
        "k1": "experiments/k1_capacity_lesion.py",
        "k2": "experiments/k2_broadcast_lesion.py",
        "k3": "experiments/k3_ignition_probe.py",
    }
    procs = {}
    for k, s in scripts.items():
        procs[k] = subprocess.run(
            [sys.executable, os.path.join(TREE, s)],
            capture_output=True, text=True, cwd=TREE, timeout=600)
    after = {k: sha256(os.path.join(TREE, v))
             for k, v in receipts.items()}
    return {k: {"exit": procs[k].returncode,
                "byte_identical": before[k] == after[k],
                "sha_before": before[k][:16],
                "sha_after": after[k][:16],
                "stderr_tail": procs[k].stderr[-500:]}
            for k in receipts}


def k4_nondegradation():
    """(c): K4 R >= 1.30 with the R1 candidate, theta=0.45."""
    wins, detail = 0, {}
    for seed in K4_SEEDS:
        tl = k4_with_tick(SubIgnitionExploreTick, seed, False)
        tf = k4_with_tick(SubIgnitionExploreTick, seed, True)
        r = tl / tf if tf > 0 else float("inf")
        win = r >= K4_MARGIN
        wins += int(win)
        detail[str(seed)] = {"learned": round(tl, 2),
                             "frozen": round(tf, 2),
                             "R": round(r, 3), "win": bool(win)}
    return wins, detail


def main():
    print("K8 — R1 wiring confirmation (preregistered; fresh seeds "
          f"{list(SEEDS)})")
    results = {"experiment": "K8_r1_wiring_confirmation",
               "seeds": list(SEEDS), "ticks": TICKS, "theta": THETA,
               "candidate": "SubIgnitionExploreTick (imported from k6)"}

    # (a1) frozen-gains ignition identity
    a1, a1n = {}, {}
    for seed in SEEDS:
        ident, nb, nr = run_frozen_identity(seed)
        a1[str(seed)] = {"ignition_trajectory_identical": bool(ident),
                         "baseline_ignited": nb, "r1_ignited": nr}
        print(f"  (a1) seed={seed}: identical={ident} "
              f"(ignited base={nb} r1={nr})")
    results["a1_frozen_ignition_identity"] = a1
    pass_a1 = all(v["ignition_trajectory_identical"] for v in a1.values())

    # (a2)/(d)/(e) learned-gains runs
    learned = {}
    for seed in SEEDS:
        learned[str(seed)] = run_learned(seed)
        b, r = learned[str(seed)]["baseline"], learned[str(seed)]["R1"]
        print(f"  seed={seed}: base ign={b['ignited']} p2c={b['phase2_c_fraction']} "
              f"tot={b['total_reward']} | R1 ign={r['ignited']} "
              f"sub_ign={r['sub_ignition_trials']} "
              f"viol={len(r['sole_path_violations'])} "
              f"p2c={r['phase2_c_fraction']} tot={r['total_reward']}")
    results["learned_runs"] = learned
    pass_d = all(len(learned[s]["R1"]["sole_path_violations"]) == 0
                 and len(learned[s]["baseline"]["sole_path_violations"]) == 0
                 for s in learned)
    pass_e = all(learned[s]["R1"]["phase2_c_fraction"] > 0.5
                 and learned[s]["R1"]["total_reward"]
                 > learned[s]["baseline"]["total_reward"]
                 and learned[s]["baseline"]["phase2_c_fraction"] < 0.1
                 for s in learned)

    # (b) K1-K3 reruns byte-identical
    bres = rerun_k1_k2_k3()
    results["b_k1k2k3_reruns"] = bres
    for k, v in bres.items():
        print(f"  (b) {k}: exit={v['exit']} "
              f"byte_identical={v['byte_identical']}")
    pass_b = all(v["exit"] == 0 and v["byte_identical"]
                 for v in bres.values())

    # (c) K4 non-degradation
    wins, detail = k4_nondegradation()
    results["c_k4_nondegradation"] = {"wins": wins, "need": K4_NEED,
                                      "margin": K4_MARGIN, "detail": detail}
    for s, d in detail.items():
        print(f"  (c) K4 seed={s}: R={d['R']} {'WIN' if d['win'] else 'loss'}")
    pass_c = wins >= K4_NEED

    gates = {"a1_frozen_ignition_identity": bool(pass_a1),
             "b_k1k2k3_byte_identical": bool(pass_b),
             "c_k4_nondegradation": bool(pass_c),
             "d_sole_path_subignition": bool(pass_d),
             "e_freeze_recovery_replicates": bool(pass_e)}
    decision = "WIRE" if all(gates.values()) else "REJECT"
    results["gates"] = gates
    results["decision"] = decision
    results["method_correction"] = (
        "First (a1) implementation compared raw item_ids and failed on "
        "seed 51502 (ws-586 vs ws-1186) despite identical ignition "
        "decisions: workspace_buffer.BoundedWorkspace._ids is a "
        "process-global itertools counter, so the second condition in "
        "the same process continues the first condition's numbering. "
        "Harness artifact, not a behavioral difference — verified by "
        "hand: same tick, same bids, same winner, one ignition each. "
        "Corrected by resetting the counter between conditions and "
        "comparing behavioral content (ignited item id+strength, "
        "competed bids, winner) per tick. The preregistration above is "
        "unchanged; this note is the honest record of the fix.")
    print("GATES:", gates)
    print("DECISION:", decision)

    out = os.path.join(TREE, "receipts", "k8_r1_wiring_confirmation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
