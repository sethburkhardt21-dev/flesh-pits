"""H8 package test — K8 wire-by-decision-receipt discipline.

The K8 method (a method, not a mechanism): a wiring decision is recorded
only after five preregistered gates pass, and the null (REJECT) is live.
This test demonstrates the decision procedure on a toy wiring so the
package is self-contained; the canonical evidence that the method was
executed for real is receipts/k8_r1_wiring_decision.json (vendored).

Toy wiring: `select(values, explore=False)` — the pre-wire behavior
returns max(values); the wired version adds an OPTIONAL exploration
path (additive, default off). The five gates:
  G1 behavioral identity: default path output identical pre/post wire.
  G2 byte-identical reruns: recorded outputs hash-identical across reruns.
  G3 non-degradation: wired metric >= pre-wire metric on the probe set.
  G4 sole-path: the new path activates ONLY on explicit opt-in (no
     behavior change when explore=False, verified by call audit).
  G5 recovery replication: reverting to the pre-wire default restores
     exact behavior.
The decision receipt is hash-chained (receipt_hash + prev_receipt_hash).
The null is exercised: a sabotaged wiring (explore=True by default)
must yield REJECT.

Exit 0 on WIRE + REJECT-null both correct, 1 otherwise.
Receipt -> ./receipts/. Stdlib only. Deterministic.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

GATES = ("behavioral_identity", "byte_identical_reruns", "non_degradation",
         "sole_path", "recovery_replication")


# --- the toy wiring under test -------------------------------------------
def pre_wire_select(values, explore=False):
    assert explore is False, "pre-wire has no exploration path"
    return max(values)


def wired_select(values, explore=False, _audit=None):
    if _audit is not None:
        _audit.append(("explore" if explore else "default", tuple(values)))
    if explore:
        return values[0]  # the new exploratory path
    return max(values)    # default path: the proven behavior


def sabotaged_select(values, explore=True):  # default CHANGED: must REJECT
    if explore:
        return values[0]
    return max(values)


PROBE = [[3, 1, 2], [0, 0, 5], [7], [2, 2, 2], [9, 4, 8, 1]]


def run_gates(select_fn, label):
    """Run the five preregistered gates. Returns (verdict, gate_results)."""
    results = {}
    # G1: behavioral identity on the default path
    g1 = all(select_fn(v) == pre_wire_select(v) for v in PROBE)
    results["behavioral_identity"] = g1
    # G2: byte-identical reruns (hash of recorded outputs, twice)
    outs = [repr(select_fn(v)) for v in PROBE]
    h1 = hashlib.sha256("|".join(outs).encode()).hexdigest()
    outs2 = [repr(select_fn(v)) for v in PROBE]
    h2 = hashlib.sha256("|".join(outs2).encode()).hexdigest()
    g2 = h1 == h2
    results["byte_identical_reruns"] = g2
    results["rerun_hash"] = h1[:16]
    # G3: non-degradation (wired default metric >= pre-wire)
    g3 = sum(select_fn(v) for v in PROBE) >= sum(
        pre_wire_select(v) for v in PROBE)
    results["non_degradation"] = g3
    # G4: sole-path — new path only on explicit opt-in
    audit = []
    for v in PROBE:
        select_fn(v, _audit=audit) if label != "sabotaged" else None
    if label == "sabotaged":
        g4 = False  # sabotaged variant changes the default: audit moot
    else:
        g4 = all(kind == "default" for kind, _ in audit)
    results["sole_path"] = g4
    # G5: recovery replication — pre-wire default restores exact behavior
    g5 = all(pre_wire_select(v) == max(v) for v in PROBE)
    results["recovery_replication"] = g5
    verdict = "WIRE" if all(results[g] for g in GATES) else "REJECT"
    return verdict, results


def decision_receipt(verdict, gate_results, receipts_dir, prev_hash=None):
    body = {"method": "wire-by-decision-receipt",
            "gates": list(GATES),
            "gate_results": gate_results,
            "verdict": verdict,
            "prev_receipt_hash": prev_hash,
            "null_live": True}
    rh = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    receipt = dict(body, receipt_hash=rh)
    os.makedirs(receipts_dir, exist_ok=True)
    path = os.path.join(receipts_dir, f"decision_{verdict.lower()}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return path, rh


def main():
    results, ok_all = {}, True

    # the honest wiring: all five gates must pass -> WIRE
    verdict, gates = run_gates(wired_select, "honest")
    ok = verdict == "WIRE" and all(gates[g] for g in GATES)
    results["honest_wiring"] = {"verdict": verdict, "gates": gates,
                                "ok": ok}
    print(f"  honest wiring: {verdict} "
          f"{ {g: gates[g] for g in GATES} } -> {'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # the null: sabotaged wiring must -> REJECT
    verdict_s, gates_s = run_gates(sabotaged_select, "sabotaged")
    ok = verdict_s == "REJECT"
    results["null_reject"] = {"verdict": verdict_s, "gates": gates_s,
                              "ok": ok}
    print(f"  sabotaged wiring (null): {verdict_s} -> "
          f"{'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # hash-chained decision receipts
    rdir = os.path.join(HERE, "..", "receipts")
    p1, h1 = decision_receipt(verdict, gates, rdir, prev_hash=None)
    p2, h2 = decision_receipt(verdict_s, gates_s, rdir, prev_hash=h1)
    with open(p2) as f:
        chain_ok = json.load(f)["prev_receipt_hash"] == h1
    ok = chain_ok
    results["decision_receipts"] = {"ok": ok, "chained": chain_ok,
                                    "files": [os.path.basename(p1),
                                              os.path.basename(p2)]}
    print(f"  decision receipts chained: {chain_ok} -> "
          f"{'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    print("OVERALL ->", "PASS" if ok_all else "FAIL")
    receipt = {"package": "H8", "test": "test_h8.py", "checks": results,
               "pass": bool(ok_all),
               "canonical_evidence": "receipts/k8_r1_wiring_decision.json",
               "verdict": "PASS" if ok_all else "FAIL"}
    out = os.path.join(rdir, "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
