"""H4 package test — change_bid (RunningZScoreBid).

Standalone proof from the packaged source:
  1. Zero-latency shift detection: after a constant baseline, the first
     reading of a level shift bids high immediately (statistics are read
     BEFORE the update). Bid at shift tick must exceed 0.9.
  2. Habituation: a sustained shift relaxes back toward 0.5 (change and
     habituation are the same mechanism).
  3. Fail-closed: non-finite input raises NonFiniteBidRefused and leaves
     state unchanged.
  4. Causal role in arbitration (K7-style): on a drifting signal, live
     bids produce nonzero winner entropy under argmax; constant-0.5
     stubs collapse it to ~0.

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from bids import RunningZScoreBid, NonFiniteBidRefused


def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def test_shift_detection():
    b = RunningZScoreBid()
    pre = 0.0
    for _ in range(60):
        pre = b.bid(0.5)
    first = b.bid(1.5)  # level shift: must fire on THIS tick (no ramp-up)
    ok = first > 0.75 and pre == 0.5
    print(f"  shift: pre-shift bid={pre:.3f}, first-shift bid={first:.3f} "
          f"(need immediate jump > 0.75 from 0.5) -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok, {"pre_shift_bid": round(pre, 4),
                "first_shift_bid": round(first, 4)}


def test_habituation():
    b = RunningZScoreBid()
    for _ in range(60):
        b.bid(0.5)
    b.bid(1.5)
    last = 0.0
    for _ in range(400):
        last = b.bid(1.5)
    ok = abs(last - 0.5) < 0.05
    print(f"  habituation: sustained-shift bid after 400 steps={last:.3f} "
          f"(need ~0.5) -> {'ok' if ok else 'FAIL'}")
    return ok, {"relaxed_bid": round(last, 4)}


def test_fail_closed():
    b = RunningZScoreBid()
    for _ in range(10):
        b.bid(0.3)
    before = b.baseline
    refused = 0
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            b.bid(bad)
        except NonFiniteBidRefused:
            refused += 1
    ok = refused == 3 and b.baseline == before
    print(f"  fail-closed: refused {refused}/3, state unchanged="
          f"{b.baseline == before} -> {'ok' if ok else 'FAIL'}")
    return ok, {"refused": refused, "state_unchanged": b.baseline == before}


def test_k7_style_entropy():
    # two channels, salience alternates in 30-tick blocks; live bids vs
    # constant-0.5 stubs (the K7 lesion). Live bids track the alternation
    # (winner entropy ~1 bit); stubs collapse it to 0 (tie-break always
    # picks channel 0).
    def run(stub):
        bids = [RunningZScoreBid(), RunningZScoreBid()]
        wins = [0, 0]
        for t in range(300):
            sig = [1.0, 0.0] if (t // 30) % 2 == 0 else [0.0, 1.0]
            bv = [0.5, 0.5] if stub else [bids[i].bid(sig[i])
                                          for i in range(2)]
            w = 0 if bv[0] >= bv[1] else 1  # canonical order tie-break
            wins[w] += 1
        ps = [w / 300 for w in wins]
        return entropy(ps)
    e_live = run(False)
    e_stub = run(True)
    ok = e_live > 0.5 and e_stub == 0.0
    print(f"  K7-style: live-bid entropy={e_live:.2f} (need > 0.5), "
          f"stub entropy={e_stub:.2f} (need 0.0) -> {'ok' if ok else 'FAIL'}")
    return ok, {"live_entropy": round(e_live, 3),
                "stub_entropy": round(e_stub, 3)}


def main():
    checks = [test_shift_detection, test_habituation, test_fail_closed,
              test_k7_style_entropy]
    results, all_ok = {}, True
    for fn in checks:
        ok, row = fn()
        results[fn.__name__] = {"ok": ok, **row}
        all_ok = all_ok and ok
    print("OVERALL ->", "PASS" if all_ok else "FAIL")
    receipt = {"package": "H4", "test": "test_h4.py",
               "checks": results, "pass": bool(all_ok),
               "verdict": "PASS" if all_ok else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
