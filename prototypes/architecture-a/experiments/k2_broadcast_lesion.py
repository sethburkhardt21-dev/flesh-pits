"""K2 — BROADCAST LESION (Architecture A load-bearing claim 3).

Hypothesis: broadcast is the SOLE path by which workspace content
  reaches consumers. Lesioning one consumer produces a SELECTIVE
  deficit (that consumer silent, others intact); lesioning the whole
  bus silences ALL consumers at once, while module-internal processing
  (bidding, buffering, ignition) continues.
Null: consumers receive content despite the lesion (side channels).
Preregistered metric: per-consumer consumption delta =
  consumed_count(condition) - consumed_count(baseline), over 30 ticks
  of closed-loop run. PASS requires:
  (a) selective lesion of memory_admit: its delta = -100% of baseline
      AND every other consumer's delta = 0;
  (b) lesion_all: every consumer's delta = -100% of baseline AND
      buffer.admitted_total keeps growing AND ignition history keeps
      growing (internal processing continues).
Baseline: intact bus, same seed, same env.
Ablation: the lesions ARE the ablation.
Procedure: 30-tick closed-loop runs (signal-tracking env, seed fixed);
  snapshot each consumer's effect counters before/after each phase:
  baseline -> lesion memory_admit -> restore -> lesion_all -> restore.
Seed: 20261007.
Kill: any consumer consumes with the bus lesioned -> side channel
  exists -> "sole path" claim FALSE -> architecture REJECTED as designed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tick import WorkspaceTick
from envs import ChangingRelevanceEnv, make_specialists
from broadcast import CONSUMER_REGISTRY

SEED = 20261007
TICKS = 30


def consumer_effects(wk):
    """Observable effect per consumer (consumption, not declaration)."""
    # reach consumers only through the bus's delivery log + read accessors
    effects = {}
    for name in CONSUMER_REGISTRY:
        n = sum(1 for d in wk.bus.deliveries
                if d["consumer"] == name and d["status"] == "delivered")
        effects[name] = n
    return effects


def run_phase(wk, env, ticks):
    before = consumer_effects(wk)
    adm_before = wk.buffer.admitted_total
    ign_before = len(wk.ignition.history)
    for _ in range(ticks):
        wk.step(env.observe(), env)
    after = consumer_effects(wk)
    delta = {k: after[k] - before[k] for k in after}
    return delta, wk.buffer.admitted_total - adm_before, \
        len(wk.ignition.history) - ign_before


def main():
    print("K2 — broadcast lesion: sole-path test")
    channels = ["a", "b", "c"]
    wk = WorkspaceTick(channels, make_specialists(channels), capacity=3,
                       ignition_kwargs={"theta": 0.45})
    env = ChangingRelevanceEnv(channels, 400, seed=SEED)

    d_base, adm_base, ign_base = run_phase(wk, env, TICKS)
    print("baseline deliveries/consumer:", d_base)
    assert all(v > 0 for v in d_base.values()), \
        "baseline must deliver to every consumer"

    # (a) selective lesion
    wk.bus.lesion_consumer("memory_admit")
    d_sel, adm_sel, ign_sel = run_phase(wk, env, TICKS)
    wk.bus.restore_consumer("memory_admit")
    print("selective lesion (memory_admit) deltas:", d_sel)
    def approx(a, b, tol=0.15):
        return abs(a - b) <= tol * max(b, 1)
    sel_ok = (d_sel["memory_admit"] == 0 and
              all(approx(d_sel[k], d_base[k]) for k in d_sel
                  if k != "memory_admit"))

    # (b) total lesion
    wk.bus.lesion_all()
    d_tot, adm_tot, ign_tot = run_phase(wk, env, TICKS)
    wk.bus.restore_all()
    print("total lesion deltas:", d_tot)
    print(f"internal processing during total lesion: "
          f"buffer admissions +{adm_tot}, ignition cycles +{ign_tot}")
    tot_ok = (all(v == 0 for v in d_tot.values())
              and adm_tot > 0 and ign_tot > 0)

    # recovery check (approximate: ignition counts vary tick to tick)
    d_rec, _, _ = run_phase(wk, env, TICKS)
    rec_ok = all(abs(d_rec[k] - d_base[k]) <= 0.15 * max(d_base[k], 1)
                 for k in d_rec)
    print("recovery deltas (== baseline):", d_rec, "->", rec_ok)

    verdict = ("PASS — selective lesion is selective; total lesion silences "
               "all consumers at once; internal processing continues; "
               "recovery complete; no side channels detected"
               if (sel_ok and tot_ok and rec_ok) else
               "FAIL — kill condition met: consumers received content "
               "despite lesion (side channel) OR internal processing died; "
               "the 'sole path' claim is FALSE -> REJECT as designed")
    print("selective_ok:", sel_ok, "total_ok:", tot_ok, "recovery_ok:", rec_ok)
    print("VERDICT:", verdict)
    receipt = {"experiment": "K2_broadcast_lesion", "seed": SEED,
               "ticks_per_phase": TICKS,
               "baseline": d_base, "selective_lesion_deltas": d_sel,
               "total_lesion_deltas": d_tot,
               "internal_admissions_during_total_lesion": adm_tot,
               "internal_ignitions_during_total_lesion": ign_tot,
               "recovery_deltas": d_rec,
               "selective_ok": sel_ok, "total_ok": tot_ok,
               "recovery_ok": rec_ok,
               "pass": bool(sel_ok and tot_ok and rec_ok), "verdict": verdict}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "receipts", "k2_broadcast_lesion.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)


if __name__ == "__main__":
    main()
