"""EXP-FP-0126 + EXP-FP-0127 receipt writer (harness-only process).

Reads the staged results (var/ecr-detector-<env>-results.json,
var/ecr-guard-grid_world-results.json) and writes two hash-chained
receipts via experiments/harness.py write_receipt, then runs
verify_chain. Runs in its own process so the harness's
experiments/envs import domain never collides with architecture-a's
(same split as write_receipt_012x.py / write_receipt_0080.py).

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


def load(var_name):
    with open(os.path.join(FLESH, "var", var_name)) as f:
        return json.load(f)


grid = load("ecr-detector-grid_world-results.json")
home = load("ecr-detector-delayed_reward-results.json")
guard = load("ecr-guard-grid_world-results.json")

assert grid["experiment_id"] == "EXP-FP-0126"
assert home["experiment_id"] == "EXP-FP-0126"
assert guard["experiment_id"] == "EXP-FP-0127"

g_ps = grid["summary"]["per_seed"]
h_ps = home["summary"]["per_seed"]
u_ps = guard["summary"]["per_seed"]

TEXT_0126 = {
    "hypothesis": ("H1: replacing ECR's hardcoded terminal detector "
                   "(r >= 1.0) with a scale-relative detector -- terminal "
                   "iff r > 0 AND r >= max(0.1, 0.5 * r_big), r_big = "
                   "running max of positive per-tick rewards -- restores "
                   "the episodic lock-in on grid_world (goal tick 0.98 >= "
                   "0.1 fires, where 0.98 < 1.0 never fired) AND preserves "
                   "the home-env behavior on canonical delayed_reward "
                   "(shaping 0.02 < 0.1 stays shaping; terminal 1.0 fires)"),
    "null": ("H0: the adaptive detector does not produce a seed-consistent "
             "learned>frozen goal-episode advantage on grid_world on 4/4 "
             "seeds, OR it regresses the home env (R < 1.30 on >= 1 "
             "delayed_reward seed)"),
    "preregistered_metric": ("grid_world: goal_episodes(learned_D3) vs "
               "goal_episodes(frozen_D3) per seed, WIN iff learned > frozen "
               "(strict), 4/4 fresh seeds {88101..88104}, 8 eps x <=100 "
               "steps (R-gate invalid: negative per-tick rewards invert "
               "the ratio -- preregistered adaptation). delayed_reward: "
               "R = total(learned_D3)/total(frozen_D3) >= 1.30 per seed, "
               "4/4 fresh seeds {88201..88204}, 8 eps x <=15 steps. "
               "H1 SUPPORTED iff both gates fire."),
    "baseline": ("frozen = identical AdaptiveTerminalECR class, "
                 "frozen=True, gains pinned 1.0, paired per seed"),
    "conditions": ("grid_world v1.0.0 + delayed_reward v1.0.0; arch_a "
                   "WorkspaceTick, neutral specialists, capacity=4, "
                   "gain_lr=0.15, theta=0.45, gain_cap=2.0; branch priors "
                   "given (grid_world: all movement channels; "
                   "delayed_reward: {branch_a,branch_b}); ECR canonical "
                   "module UNTOUCHED (additive subclass only); prereg "
                   "experiments/preregistration_EXP-FP-0126.json"),
    "interpretation": (
        "H0 -- CLEAN KILL. grid_world: BREAKS 0/4. D3 learned goals "
        + "/".join(f"{p['learned_goal_episodes']}" for p in g_ps)
        + " vs frozen "
        + "/".join(f"{p['frozen_goal_episodes']}" for p in g_ps)
        + " (per-seed totals: "
        + ", ".join(f"{p['learned_total']:.2f}/{p['frozen_total']:.2f}"
                    for p in g_ps)
        + "). The detector fix works MECHANICALLY: gain-history "
          "inspection (seed 88101) shows the 0.98 goal tick flagged "
          "terminal=True, firing the lock-in -- branch_credit=north "
          "(first branch winner) + progress_credit=north (canonical "
          "tie-break, zero shaping receipts). Final gains {north: 2.0, "
          "rest 1.0}: the agent became a north-attractor with no "
          "unlearning path (non-goal ticks are negative/zero and change "
          "no gains); learned totals more negative than frozen on 3/4 "
          "seeds. The lock-in is actively harmful where the progress "
          "channel is unidentifiable from the reward stream. "
          "delayed_reward: HOLDS 4/4, R = "
        + "/".join(f"{p['ratio']:.2f}" for p in h_ps)
        + " -- the home env is NOT regressed (no tick falls in "
          "(0.1, 1.0), so D3 is behaviorally identical to D0 there). "
          "Ablations (descriptive, first 2 seeds): D0 learned is "
          "bit-identical to frozen on grid_world (reproduces the 0120 "
          "failure exactly); D1 and D2 are gain-identical to D3 on both "
          "envs -- the floor does all the work, relativity adds nothing "
          "on these reward streams. The 0120 bound is REFINED, not "
          "lifted: even with a working terminal detector, ECR's episodic "
          "lock-in requires a reward stream from which the progress "
          "channel is identifiable."),
    "limitations": ("4 seeds/env; neutral specialists (bandit isolation); "
                    "branch_channels priors given; the floor 0.1 is a "
                    "scale prior, not learned; first 0126 run declared "
                    "VOID on a G2 driver bug (tuple unpack) and rerun "
                    "deterministically -- G2 exact-match confirms the "
                    "numbers."),
}

TEXT_0127 = {
    "hypothesis": ("H1: disarming the arbitrary progress cap-set "
                   "(branch-only lock-in when the episode contains zero "
                   "shaping receipts) removes the harm measured in "
                   "EXP-FP-0126: D3+guard learned gains show no "
                   "arbitrary-channel pinning, and learned is no longer "
                   "worse than frozen on grid_world"),
    "null": ("H0: the harm persists under the guard (learned goals < "
             "frozen goals on >= 1 seed) -- the branch pin alone recreates "
             "the attractor, and the lock-in is inherently mismatched to "
             "navigation tasks"),
    "preregistered_metric": ("grid_world: goal_episodes(learned) vs "
               "goal_episodes(frozen) per seed; NO-HARM iff learned >= "
               "frozen (not strict) on 4/4 fresh seeds {88301..88304}, "
               "8 eps x <=100 steps. Bit-identity check: delayed_reward "
               "seeds 88201/88202 reused -- D3+guard learned totals must "
               "equal the EXP-FP-0126 D3 totals exactly (guard inert where "
               "shaping exists)."),
    "baseline": ("frozen = same class (D3 + guard), frozen=True, gains "
                 "pinned 1.0, paired per seed"),
    "conditions": ("grid_world v1.0.0; arch_a WorkspaceTick, neutral "
                   "specialists, capacity=4, gain_lr=0.15, theta=0.45, "
                   "gain_cap=2.0; branch prior = all movement channels; "
                   "detector D3 (FLOOR=0.1, REL=0.5) + progress_guard=True; "
                   "canonical files UNTOUCHED; prereg experiments/"
                   "preregistration_EXP-FP-0127.json"),
    "interpretation": (
        "H0 -- HARM-PERSISTS (3/4 no-harm). Per-seed (learned/frozen "
        "goals): "
        + ", ".join(f"{p['learned_goal_episodes']}/{p['frozen_goal_episodes']}"
                    for p in u_ps)
        + ". The guard works as designed (terminal ticks record "
          "progress_credit=None; no arbitrary channel capped), but on "
          "seed 88302 the branch pin alone (north=2.0, the first-branch "
          "winner) recreated the one-direction attractor: goals 1 vs "
          "frozen 2. On 88301 the branch pin (south=2.0) was "
          "goal-neutral (1/11 = 1/11); 88303/88304 never reached a goal "
          "so the guard is trivially identical to frozen. The tie-break "
          "is NOT the sole harmful half: pinning ANY single movement "
          "channel at the gain cap on a navigation task builds an "
          "attractor, because (a) every movement tick counts as a "
          "'branch' decision under the given prior, and (b) no "
          "unlearning path exists for negative/zero ticks. Bit-identity "
          "check: D3+guard == D3 exactly on delayed_reward (3.080000 / "
          "1.980000 MATCH) -- the guard is provably inert where shaping "
          "exists. No further follow-up in this lane (preregistered "
          "stop). The refined ECR bound: the episodic lock-in is a "
          "corridor-task mechanism; on navigation tasks it is not merely "
          "inert but actively harmful."),
    "limitations": ("4 seeds; neutral specialists; the no-harm gate is "
                    "weaker than a win gate by design -- this experiment "
                    "characterizes which half of the lock-in carries the "
                    "harm; it does not promise ECR wins on navigation."),
}


def main():
    receipts_dir = os.path.join(FLESH, "receipts")
    out = write_receipt(
        {"experiment_id": "EXP-FP-0126",
         "config": grid["config"],
         "config_hash": grid["config_hash"],
         "primary_seed": grid["primary_seed"],
         "started_utc": grid["started_utc"],
         "episodes": [],
         "summary": {
             "n_episodes": 0,
             "mean_return": 0.0, "stdev_return": 0.0,
             "min_return": 0.0, "max_return": 0.0,
             "per_seed": g_ps,
             "wins": grid["summary"]["wins"],
             "need": grid["summary"]["need"],
             "margin": 1.30,
             "passed": grid["summary"]["passed"],
             "verdict": grid["summary"]["verdict"],
             "delayed_reward_gate": {
                 "per_seed": h_ps,
                 "wins": home["summary"]["wins"],
                 "verdict": home["summary"]["verdict"],
             },
             "ablations": grid["summary"]["ablations"],
         }},
        receipts_dir, **{k: v for k, v in TEXT_0126.items()})
    print("receipt:", out)
    out = write_receipt(
        {"experiment_id": "EXP-FP-0127",
         "config": guard["config"],
         "config_hash": guard["config_hash"],
         "primary_seed": guard["primary_seed"],
         "started_utc": guard["started_utc"],
         "episodes": [],
         "summary": guard["summary"]},
        receipts_dir, **{k: v for k, v in TEXT_0127.items()})
    print("receipt:", out)
    ok, problems = verify_chain(receipts_dir)
    print(f"verify_chain: {'OK' if ok else 'PROBLEMS: ' + str(problems)}")
    if not ok:
        ours = [p for p in problems if "EXP-FP-0126" in p
                or "EXP-FP-0127" in p]
        if ours:
            raise SystemExit(f"OUR receipts have chain problems: {ours}")
        print("chain problems are pre-existing/historical only; "
              "EXP-FP-0126/0127 receipts chain cleanly.")


if __name__ == "__main__":
    main()
