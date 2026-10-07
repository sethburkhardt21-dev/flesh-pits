"""EXP-FP-0080 receipt writer (harness-only process).

Reads the staged driver results (var/ecr-repro-0080-results.json) and
writes the hash-chained receipt via experiments/harness.py write_receipt,
then runs verify_chain. Runs in its own process so the harness's
experiments/envs import domain never collides with architecture-a's.

If the harness raises ReceiptExistsError (fail-closed on existing
paths), this script does NOT force: it exits non-zero and the operator
claims a fresh ID.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLESH = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(FLESH, "experiments"))

from harness import write_receipt, verify_chain  # noqa: E402

EXPERIMENT_ID = "EXP-FP-0080"


def main():
    results_path = os.path.join(FLESH, "var", "ecr-repro-0080-results.json")
    with open(results_path) as f:
        result = json.load(f)
    assert result["experiment_id"] == EXPERIMENT_ID, "staged ID mismatch"
    passed = result["summary"]["passed"]

    path = write_receipt(
        result, os.path.join(FLESH, "receipts"),
        hypothesis=("H1: an independent implementation of the ECR gain "
                    "update (return-conditioned, baseline-free, episodic) "
                    "beats frozen gains on canonical delayed_reward "
                    "(R >= 1.30 on 4/4 fresh seeds)"),
        null=("H0: R < 1.30 on >= 1 seed -> replication fails -> "
              "negative result with mechanism account"),
        preregistered_metric="R = total(learned)/total(frozen) per seed, "
                             "8 episodes x <=15 steps; per-seed WIN iff "
                             "R >= 1.30",
        baseline="frozen = identical ReturnConditionedEpisodicArbitrator "
                 "with frozen=True (gains pinned at 1.0), paired per seed",
        conditions=("delayed_reward v1.0.0; neutral specialists; capacity=4; "
                    "gain_lr=0.15; theta=0.45; gain_cap=2.0; independent "
                    "module prototypes/architecture-a/"
                    "attention_ecr_repro.py; preregistration "
                    "experiments/preregistration_EXP-FP-0080.json; "
                    "driver prototypes/architecture-a/experiments/"
                    "exp_fp_0080_ecr_repro.py"),
        interpretation=(
            "REPRODUCED: independent ECR implementation lifts the "
            "NR-A-006 bound on delayed_reward (4/4 seeds)" if passed else
            "FAIL: independent ECR does not replicate EXP-FP-0021; "
            "mechanism account required"),
        limitations=("4 seeds; single env (canonical delayed_reward); "
                     "task-structural priors (branch_channels, max_steps) "
                     "given, not learned; per-episode (not cumulative) "
                     "shaping tally is the independent re-derivation "
                     "choice -- equivalent on this env where only "
                     "forward earns shaping"),
    )
    print("receipt:", path)

    ok, problems = verify_chain(os.path.join(FLESH, "receipts"))
    print(f"verify_chain: {'OK' if ok else 'PROBLEMS: ' + str(problems)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
