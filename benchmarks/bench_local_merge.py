#!/usr/bin/env python3
"""Merge per-rung battery JSONs into benchmarks/bench_local_1.5b_full.json:
full run headers + results + 1.5B-vs-0.5B battery table. Run after both rungs."""
import json, os, time
FP = os.path.expanduser("~/workspace/chambers/emergent-mind/flesh-pits")
BM = os.path.join(FP, "benchmarks")

def load(p):
    with open(p) as f: return json.load(f)

r1 = load(os.path.join(BM, "bench_local_battery_rung1.json"))
r2 = load(os.path.join(BM, "bench_local_battery_rung2.json"))

DIM_ORDER = ["cognition", "instruction_following", "structured_output", "tool_use",
             "long_context", "latency", "contamination", "reproducibility"]

def summarize(dim):
    items = dim.get("items", [])
    return {"item_verdicts": [i.get("verdict") for i in items]}

table = []
for d in DIM_ORDER:
    d1, d2 = r1["dimensions"][d], r2["dimensions"][d]
    table.append({
        "dimension": d,
        "rung1_1.5b": {"verdict": d1.get("dimension_verdict"), "summary": summarize(d1)},
        "rung2_0.5b": {"verdict": d2.get("dimension_verdict"), "summary": summarize(d2)},
    })

out = {
    "schema": "emergent-mind/section19-battery/1",
    "experiment_ids": ["BENCH-1.5B-FULL-BATTERY", "BENCH-0.5B-BASELINE-BATTERY"],
    "preregistration": "benchmarks/bench_local_battery_preregistration.json",
    "t_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "backend_change_events": "NONE — both rungs used LocalLlamaCpp (llama-cpp-python 0.3.36); no backend change, no run-ID minting",
    "runs": {"rung1_1.5b": r1, "rung2_0.5b": r2},
    "battery_table": table,
}
p = os.path.join(BM, "bench_local_1.5b_full.json")
with open(p, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", p)
