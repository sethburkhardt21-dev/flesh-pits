"""Phase-4 5-seed multi-seed replication — Architecture B (§9 item #3).

ADDITIVE driver. The instruments in experiments_phase4.py (exp_m1, exp_c2b,
exp_c4b, exp_k3b) are reused VERBATIM — no redefinition, no parameter changes,
no logic changes. The ONLY changes are:

  1. the seed-list parameter (3 original seeds -> 5 NEW fresh seeds), applied
     by assigning a modified deepcopy to the module's <NAME>_PREREG dict,
  2. the receipt ID ("EXP-AB-<X>" -> "repro5_EXP-AB-<X>"), applied by wrapping
     the module's write_receipt lookup in the module namespace.

Both patches are restored after each run (try/finally). The original receipts
are never written to.

Receipts follow the EXP-AB-K3C precedent:
  - arch-b detail receipt: experiments.write_receipt output, then
    prev_receipt_hash = sha256 of the ORIGINAL EXP-AB-<X>.json file bytes,
    receipt_hash over the canonical body (same chaining construction as K3C).
  - lane summary receipt: harness.write_receipt into the lane receipts/ dir
    (joins the harness hash chain).

Replication verdicts use the preregistered per-seed rules in
experiments/preregistration_EXP-AB-5SEED.json (>= 4/5 holds).

Usage:
    python3 repro_phase4_5seed.py [--exp M1|C2B|C4B|K3B]
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FP_EXP = os.path.abspath(os.path.join(HERE, "..", "..", "experiments"))
sys.path.insert(0, FP_EXP)

import experiments_phase4 as p4  # noqa: E402
from experiments import write_receipt as _orig_write_receipt  # noqa: E402
from experiments import (  # noqa: E402
    ENVS_BY_NAME, make_agent, run_closed_loop, _mean,
)

ARCH_RECEIPTS = os.path.join(HERE, "receipts")
LANE_RECEIPTS = os.path.abspath(os.path.join(FP_EXP, "..", "receipts"))

# Fresh seeds — never used by any prior experiment (see preregistration
# fresh_seed_inventory; checked 2026-10-07 across the registry, all receipts,
# and all driver files).
SEEDS = {
    "M1": [75101, 75102, 75103, 75104, 75105],
    "C2B": [75201, 75202, 75203, 75204, 75205],
    "C4B": [75301, 75302, 75303, 75304, 75305],
    "K3B": [75401, 75402, 75403, 75404, 75405],
}

REPRO_ID = {
    "M1": "repro5_EXP-AB-M1",
    "C2B": "repro5_EXP-AB-C2B",
    "C4B": "repro5_EXP-AB-C4B",
    "K3B": "repro5_EXP-AB-K3B",
}

ORIG_ID = {
    "M1": "EXP-AB-M1",
    "C2B": "EXP-AB-C2B",
    "C4B": "EXP-AB-C4B",
    "K3B": "EXP-AB-K3B",
}

PREREG_ATTR = {
    "M1": "M1_PREREG",
    "C2B": "C2B_PREREG",
    "C4B": "C4B_PREREG",
    "K3B": "K3B_PREREG",
}

EPS = 1e-9


def _remapped_write_receipt(orig_id, new_id):
    def _wrap(exp_id, prereg, result, interpretation, limitations):
        return _orig_write_receipt(
            new_id if exp_id == orig_id else exp_id,
            prereg, result, interpretation, limitations)
    return _wrap


def _chain_arch_receipt(path, orig_receipt_path):
    """K3C-style chain link: prev = sha256 of the original receipt's bytes."""
    with open(path) as f:
        body = json.load(f)
    with open(orig_receipt_path, "rb") as f:
        prev_hash = hashlib.sha256(f.read()).hexdigest()
    body["prev_receipt_hash"] = prev_hash
    canonical = json.dumps(body, indent=2, sort_keys=True)
    body["receipt_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    with open(path, "w") as f:
        json.dump(body, f, indent=2, sort_keys=True)


def run_experiment(name):
    """Run one Phase-4 instrument verbatim on the 5 fresh seeds."""
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prereg_attr = PREREG_ATTR[name]
    saved_prereg = getattr(p4, prereg_attr)
    saved_write = p4.write_receipt
    prereg = copy.deepcopy(saved_prereg)
    prereg["seeds"] = list(SEEDS[name])
    prereg["conditions"]["replication_of"] = ORIG_ID[name]
    prereg["conditions"]["reproduction_seeds"] = list(SEEDS[name])
    setattr(p4, prereg_attr, prereg)
    p4.write_receipt = _remapped_write_receipt(ORIG_ID[name], REPRO_ID[name])
    try:
        out, bad = p4.EXPS[name]()
    finally:
        setattr(p4, prereg_attr, saved_prereg)
        p4.write_receipt = saved_write
    # chain-link the arch-b detail receipt (K3C precedent)
    arch_path = os.path.join(ARCH_RECEIPTS, REPRO_ID[name] + ".json")
    _chain_arch_receipt(arch_path, os.path.join(ARCH_RECEIPTS,
                                                ORIG_ID[name] + ".json"))
    return out, bad, started, arch_path


# ---------------------------------------------------------------------------
# Replication verdicts (preregistered per-seed rules)
# ---------------------------------------------------------------------------

def verdict_m1(out):
    pom = {p["seed"]: p["delta"] for p in out["pomaze"]["per_seed"]}
    drw = {p["seed"]: p["delta"] for p in out["delayed_reward"]["per_seed"]}
    holds = [s for s in SEEDS["M1"]
             if pom[s] > 0 and pom[s] > drw[s] + EPS]
    return holds, len(holds)


def verdict_c2b(out):
    holds = []
    for i, s in enumerate(SEEDS["C2B"]):
        sh, st = out["shift"], out["stationary"]
        revival = (sh["shift_reset"]["per_seed_returns"][i]
                   >= sh["estimated"]["per_seed_returns"][i] - EPS
                   and sh["shift_reset"]["per_seed_returns"][i]
                   > sh["uniform"]["per_seed_returns"][i] + EPS
                   and st["shift_reset"]["per_seed_returns"][i]
                   >= st["uniform"]["per_seed_returns"][i] - EPS)
        if revival:
            holds.append(s)
    return holds, len(holds)  # holds = seeds where revival SUCCEEDED


def verdict_c4b(out):
    holds = []
    for i, s in enumerate(SEEDS["C4B"]):
        ai = out["active_inference"]["per_seed"][i]["IG_probe"]
        gr = out["greedy"]["per_seed"][i]["IG_probe"]
        ra = out["random"]["per_seed"][i]["IG_probe"]
        if ai <= max(gr, ra) + EPS:
            holds.append(s)  # holds = seeds where the kill condition holds
    return holds, len(holds)


def verdict_k3b(out):
    holds = []
    for i, s in enumerate(SEEDS["K3B"]):
        p = out["arm_a_changing_rule"]["per_seed"][i]
        if p["delta_e0"] > 0.0032 + EPS or p["delta_rerr"] > 0.0032 + EPS:
            holds.append(s)
    return holds, len(holds)


VERDICT_FN = {
    "M1": (verdict_m1, "selective deficit",
           "REPRODUCES", "FRAGILE", "OVERTURNED"),
    "C2B": (verdict_c2b, "revival",
            "REVIVAL", "REVIVAL (partial)", "KILL REPLICATES"),
    "C4B": (verdict_c4b, "kill condition",
            "KILL REPLICATES", "KILL FRAGILE", "KILL OVERTURNED"),
    "K3B": (verdict_k3b, "hierarchy grown",
            "REPRODUCES", "FRAGILE", "OVERTURNED"),
}


def interpret_replication(name, out, n_holds):
    fn, what, v_high, v_mid, v_low = VERDICT_FN[name]
    if n_holds >= 4:
        v = v_high
    elif n_holds == 3:
        v = v_mid
    else:
        v = v_low
    return (f"5-SEED REPLICATION ({n_holds}/5 seeds: {what}): {v}. "
            f"Recorded verdict was on 3 seeds; this replication ran the "
            f"identical instrument on 5 fresh seeds "
            f"{SEEDS[name]}.")


# ---------------------------------------------------------------------------
# G0a determinism spot-check: re-run M1 seed 75101 through the identical
# helper calls and compare to the recorded per-seed numbers at 1e-12.
# ---------------------------------------------------------------------------

def check_determinism_m1(out):
    s = SEEDS["M1"][0]
    rec_pom = next(p for p in out["pomaze"]["per_seed"] if p["seed"] == s)
    rec_drw = next(p for p in out["delayed_reward"]["per_seed"]
                   if p["seed"] == s)
    got = {}
    for env_name, n_eps in (("pomaze", 15), ("delayed_reward", 10)):
        for mem in (True, False):
            a = make_agent(ENVS_BY_NAME[env_name], seed=1000 + s,
                           affect="none", memory_enabled=mem)
            eps = run_closed_loop(env_name, a, n_eps, s)
            got[(env_name, mem)] = _mean([e["return"] for e in eps])
    checks = {
        "pomaze_intact": (rec_pom["intact"], got[("pomaze", True)]),
        "pomaze_no_memory": (rec_pom["no_memory"], got[("pomaze", False)]),
        "delayed_intact": (rec_drw["intact"], got[("delayed_reward", True)]),
        "delayed_no_memory": (rec_drw["no_memory"],
                              got[("delayed_reward", False)]),
    }
    bad = {k: (r, g) for k, (r, g) in checks.items() if abs(r - g) > 1e-12}
    return (len(bad) == 0, checks, bad)


# ---------------------------------------------------------------------------
# Lane summary receipts (harness hash chain)
# ---------------------------------------------------------------------------

def lane_summary(name, out, holds, verdict_line, det_line, arch_path):
    if name == "M1":
        summary = {
            "verdict": verdict_line,
            "seeds": SEEDS[name],
            "per_seed": [
                {"seed": p["seed"],
                 "pomaze_delta": next(q["delta"] for q in
                                      out["pomaze"]["per_seed"]
                                      if q["seed"] == p["seed"]),
                 "delayed_reward_delta": next(q["delta"] for q in
                                              out["delayed_reward"]
                                              ["per_seed"]
                                              if q["seed"] == p["seed"]),
                 "selective": p["seed"] in holds}
                for p in out["pomaze"]["per_seed"]],
            "seed_mean_delta_pomaze": out["pomaze"]["mean_delta"],
            "seed_mean_delta_delayed_reward":
                out["delayed_reward"]["mean_delta"],
            "detail_receipt": os.path.relpath(arch_path, os.path.join(
                FP_EXP, "..")),
        }
    elif name == "C2B":
        summary = {
            "verdict": verdict_line,
            "seeds": SEEDS[name],
            "revival_holds_seeds": holds,
            "shift": {m: {"seed_mean_return": out["shift"][m]["mean_return"],
                          "per_seed": out["shift"][m]["per_seed_returns"]}
                      for m in ("uniform", "estimated", "shift_reset")},
            "stationary": {m: {"seed_mean_return":
                               out["stationary"][m]["mean_return"],
                                "per_seed":
                                out["stationary"][m]["per_seed_returns"]}
                           for m in ("uniform", "estimated", "shift_reset")},
            "detail_receipt": os.path.relpath(arch_path, os.path.join(
                FP_EXP, "..")),
        }
    elif name == "C4B":
        summary = {
            "verdict": verdict_line,
            "seeds": SEEDS[name],
            "kill_holds_seeds": holds,
            "modes": {
                m: {"seed_mean_IG_probe": out[m]["mean_IG_probe"],
                    "seed_mean_return": out[m]["mean_return"],
                    "per_seed_IG_probe": [p["IG_probe"]
                                          for p in out[m]["per_seed"]]}
                for m in ("active_inference", "greedy", "random")},
            "detail_receipt": os.path.relpath(arch_path, os.path.join(
                FP_EXP, "..")),
        }
    else:  # K3B
        summary = {
            "verdict": verdict_line,
            "seeds": SEEDS[name],
            "grown_seeds": holds,
            "arm_a": {
                "seed_mean_delta_e0":
                    out["arm_a_changing_rule"]["mean_delta_e0"],
                "seed_mean_delta_rerr":
                    out["arm_a_changing_rule"]["mean_delta_rerr"],
                "per_seed": [
                    {"seed": p["seed"], "delta_e0": p["delta_e0"],
                     "delta_rerr": p["delta_rerr"],
                     "grown": p["seed"] in holds}
                    for p in out["arm_a_changing_rule"]["per_seed"]]},
            "arm_b": {
                "seed_mean_delta_e0":
                    out["arm_b_delayed_reward"]["mean_delta_e0"],
                "seed_mean_delta_rerr_terminal":
                    out["arm_b_delayed_reward"]
                    ["mean_delta_rerr_terminal"]},
            "detail_receipt": os.path.relpath(arch_path, os.path.join(
                FP_EXP, "..")),
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL",
                    choices=sorted(SEEDS) + ["ALL"])
    args = ap.parse_args(argv)
    names = sorted(SEEDS) if args.exp == "ALL" else [args.exp]
    sys.path.insert(0, FP_EXP)
    from harness import write_receipt as lane_write_receipt  # noqa: E402
    from env_interface import config_hash  # noqa: E402
    prereg_doc = json.load(open(os.path.join(
        FP_EXP, "preregistration_EXP-AB-5SEED.json")))
    for name in names:
        print(f"=== repro5_EXP-AB-{name} ===", flush=True)
        out, _bad, started, arch_path = run_experiment(name)
        fn, _what, _vh, _vm, _vl = VERDICT_FN[name]
        holds, n_holds = fn(out)
        verdict_line = interpret_replication(name, out, n_holds)
        det_line = ""
        if name == "M1":
            ok, _checks, bad = check_determinism_m1(out)
            det_line = (f"G0a determinism spot-check (seed {SEEDS['M1'][0]}, "
                        f"identical helper calls): "
                        f"{'PASS (all 4 re-runs match to 1e-12)' if ok else 'FAIL: ' + str(bad)}")
            print("  " + det_line, flush=True)
        # append the replication verdict into the arch-b detail receipt
        with open(arch_path) as f:
            body = json.load(f)
        rep = {
            "experiment_id": REPRO_ID[name],
            "replicates": ORIG_ID[name],
            "seeds": SEEDS[name],
            "holds_seeds": holds,
            "n_holds": n_holds,
            "verdict": verdict_line,
            "determinism_g0a": det_line,
        }
        body["interpretation"] = (body.get("interpretation", "")
                                  + "\n\n" + verdict_line
                                  + ("\n" + det_line if det_line else ""))
        body["replication"] = rep
        canonical = json.dumps({k: v for k, v in body.items()
                                if k != "receipt_hash"},
                               indent=2, sort_keys=True)
        body["receipt_hash"] = hashlib.sha256(
            canonical.encode()).hexdigest()
        with open(arch_path, "w") as f:
            json.dump(body, f, indent=2, sort_keys=True)
        # lane summary receipt (joins the harness chain)
        summ = lane_summary(name, out, holds, verdict_line, det_line,
                            arch_path)
        entry = prereg_doc["experiments"][REPRO_ID[name]]
        lane_result = {
            "experiment_id": REPRO_ID[name],
            "config": {"seeds": SEEDS[name],
                       "instrument": "experiments_phase4.py verbatim "
                                     f"({ORIG_ID[name]})",
                       "replicates": ORIG_ID[name]},
            "config_hash": config_hash(entry),
            "primary_seed": "-".join(str(s) for s in
                                     (SEEDS[name][0], SEEDS[name][-1])),
            "started_utc": started,
            "episodes": {},
            "summary": summ,
        }
        lane_path = lane_write_receipt(
            lane_result, LANE_RECEIPTS,
            hypothesis=entry["hypothesis"], null=entry["null"],
            preregistered_metric=entry["metric"],
            baseline=entry["baseline"],
            conditions=entry["procedure"] + " | " + entry["pass_fail"],
            interpretation=verdict_line + ("\n" + det_line if det_line
                                           else ""),
            limitations="Instrument verbatim from Phase-4; only the seed "
                        "list changed. 5 fresh seeds; no tuning.")
        print(f"  {verdict_line}\n  arch: {arch_path}\n  lane: {lane_path}",
              flush=True)


if __name__ == "__main__":
    main()
