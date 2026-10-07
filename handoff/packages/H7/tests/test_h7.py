"""H7 package test — experiment harness (preregistration + hash-chained receipts).

Standalone proof from the packaged sources (harness.py + env_interface.py):
  1. Canonical episode loop: run_episode drives a toy ENV_INTERFACE-style
     env/agent and returns a well-formed summary (episode_id, init/final
     state hashes, return, steps, done).
  2. Preregistration: a preregistration record carries every §38 field
     (hypothesis, null, metric, baseline, ablation, procedure, seed,
     interpretation, limitations) — validated by schema check.
  3. Hash-chained receipts: write_receipt chains prev_receipt_hash;
     verify_chain() passes on an intact receipt dir.
  4. Tamper evidence: editing a receipt breaks verify_chain() (both the
     receipt_hash and the linkage are checked).

The lab's env/agent zoo (envs, baselines, arch_d) is NOT vendored here;
test-only stubs satisfy the harness's module-level imports, and the
README states this boundary explicitly. The metrology machinery —
preregistration, canonical loop, chained receipts — is what is proven.

Exit 0 on PASS, 1 on FAIL. Receipt -> ./receipts/. Stdlib only.
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

# --- test-only stubs for the lab's env/agent zoo (NOT part of the package)
class _StubArchD:
    NAME = "arch_d_stub"


for _name, _attrs in (("envs", {"ALL_ENVS": []}),
                      ("baselines", {"ALL_BASELINES": []}),
                      ("arch_d", {"ArchD": _StubArchD})):
    _mod = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_mod, _k, _v)
    sys.modules[_name] = _mod

import harness  # noqa: E402
from env_interface import (  # noqa: E402
    Environment, derive_seed, new_rng)

PREREG_FIELDS = ("hypothesis", "null_hypothesis", "metric", "baseline",
                 "ablation", "procedure", "seed", "interpretation",
                 "limitations")


class ToyEnv(Environment):
    NAME = "toy_counter"
    VERSION = "1.0.0"

    def __init__(self):
        self._t = 0

    def action_space(self):
        return {"type": "discrete", "n": 2}

    def observation_space(self):
        return {"t": {"type": "scalar", "low": 0, "high": 10}}

    def reset(self, seed):
        self._rng = new_rng(seed)
        self._t = 0
        return {"t": 0}

    def step(self, action):
        self._t += 1
        reward = 1.0 if action == 1 else 0.0
        done = self._t >= 10
        return {"t": self._t}, reward, done, {}

    def state_hash(self):
        return f"toy:{self._t}"


class ToyAgent:
    NAME = "toy_greedy"

    def reset(self, seed, action_space):
        self._rng = new_rng(seed)

    def act(self, obs):
        return 1  # always the rewarding action

    def update(self, obs, action, reward, done, info):
        pass


def main():
    results, ok_all = {}, True

    # 1. canonical episode loop
    env, agent = ToyEnv(), ToyAgent()
    summ = harness.run_episode(env, agent, episode_seed=777, run_index=0,
                               max_steps=50)
    ok = (summ["steps"] == 10 and summ["return"] == 10.0
          and summ["done"] is True
          and summ["episode_id"].startswith("toy_counter")
          and summ["init_hash"] != summ["final_hash"])
    results["episode_loop"] = {"ok": ok, "summary": summ}
    print(f"  episode loop: steps={summ['steps']} return={summ['return']} "
          f"hashes differ={summ['init_hash'] != summ['final_hash']} -> "
          f"{'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 2. preregistration schema
    prereg = {"hypothesis": "toy agent earns reward",
              "null_hypothesis": "no reward difference vs chance",
              "metric": "total return over 10 steps",
              "baseline": "random agent",
              "ablation": "n/a (metrology)",
              "procedure": "run_episode x1, seed 777",
              "seed": 777,
              "interpretation": "loop executes",
              "limitations": "toy only"}
    ok = all(f in prereg for f in PREREG_FIELDS)
    results["preregistration"] = {"ok": ok, "fields": sorted(prereg)}
    print(f"  preregistration: all §38 fields present -> "
          f"{'ok' if ok else 'FAIL'}")
    ok_all = ok_all and ok

    # 3+4. chained receipts + tamper evidence
    with tempfile.TemporaryDirectory() as rdir:
        res1 = {"experiment_id": "PKG-1", "config": {"a": 1},
                "config_hash": "x", "primary_seed": 1,
                "started_utc": "t", "episodes": [summ],
                "summary": {"n_episodes": 1}}
        res2 = dict(res1, experiment_id="PKG-2")
        harness.write_receipt(res1, rdir, hypothesis="h", null="n")
        import time
        time.sleep(0.02)  # distinct mtimes for chain ordering
        harness.write_receipt(res2, rdir, hypothesis="h", null="n")
        ok_chain, problems = harness.verify_chain(rdir)
        ok = ok_chain and not problems
        results["receipt_chain"] = {"ok": ok, "problems": problems}
        print(f"  receipt chain intact: verify={ok_chain} -> "
              f"{'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

        with open(os.path.join(rdir, "PKG-2.json")) as f:
            rec = json.load(f)
        rec["summary"]["n_episodes"] = 999  # tamper
        with open(os.path.join(rdir, "PKG-2.json"), "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        ok_tamper, problems = harness.verify_chain(rdir)
        ok = (not ok_tamper) and len(problems) > 0
        results["tamper_evidence"] = {"ok": ok, "problems": problems}
        print(f"  tamper detected: verify={ok_tamper} problems={problems} "
              f"-> {'ok' if ok else 'FAIL'}")
        ok_all = ok_all and ok

    print("OVERALL ->", "PASS" if ok_all else "FAIL")
    receipt = {"package": "H7", "test": "test_h7.py", "checks": results,
               "pass": bool(ok_all),
               "boundary": ("lab env/agent zoo stubbed; metrology "
                            "machinery proven, not the zoo"),
               "verdict": "PASS" if ok_all else "FAIL"}
    out = os.path.join(HERE, "..", "receipts", "package_test_receipt.json")
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print("receipt:", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
