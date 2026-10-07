"""EXP-FP-0120..0125 receipt writer (harness-only process).

Reads the staged per-env battery results (var/ecr-gen-<env>-results.json)
and writes six hash-chained receipts via experiments/harness.py
write_receipt, then runs verify_chain. Runs in its own process so the
harness's experiments/envs import domain never collides with
architecture-a's (same split as write_receipt_0080.py).

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

ENVS = ["grid_world", "pomaze", "changing_rule", "compositional_rule",
        "cue_delayed_reward", "delayed_multistep"]

TEXT = {
    "grid_world": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on "
                       "grid_world: learned reaches the goal in strictly "
                       "more episodes than frozen on 4/4 fresh seeds"),
        "null": ("H0: learned does not out-reach frozen on >=1 seed -> "
                 "ECR does not generalize to grid_world"),
        "metric": ("goal_episodes(learned) vs goal_episodes(frozen) per "
                   "seed; WIN iff learned > frozen (strict). R-gate "
                   "INVALID here (negative per-tick rewards invert the "
                   "ratio) -- preregistered adaptation"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "conditions": ("grid_world v1.0.0; neutral specialists; capacity=4; "
                       "gain_lr=0.15; theta=0.45; gain_cap=2.0; branch "
                       "prior = all movement channels; max_steps=100; 8 "
                       "episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0120.json"),
        "interpretation": ("BREAKS (0/4). The goal tick pays -0.02+1.00="
                           "0.98 < 1.0, so ECR's terminal-success detector "
                           "never fires: terminal success is absorbed as a "
                           "shaping tick and the success lock-in "
                           "(branch+progress cap-set) never executes. "
                           "Episode boundaries leak across env episodes "
                           "until the max_steps truncation reset. Learned "
                           "gains stay ~1.0 (tiny shaping-classified "
                           "updates on the 1-2 goal ticks reached); goal "
                           "counts tied with frozen on all 4 seeds. All "
                           "three ablations ~identical to full "
                           "(sub-mechanisms inert in this regime)."),
        "limitations": ("4 seeds; neutral specialists (bandit isolation); "
                        "branch_channels prior given; ECR module untouched "
                        "from EXP-FP-0080."),
    },
    "pomaze": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on pomaze: "
                       "learned reaches the goal in strictly more episodes "
                       "than frozen on 4/4 fresh seeds"),
        "null": ("H0: learned does not out-reach frozen on >=1 seed -> "
                 "ECR does not generalize to pomaze"),
        "metric": ("goal_episodes(learned) vs goal_episodes(frozen) per "
                   "seed; WIN iff learned > frozen (strict). R-gate "
                   "INVALID (negative per-tick rewards) -- preregistered "
                   "adaptation"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "conditions": ("pomaze v1.0.0; neutral specialists; capacity=4; "
                       "gain_lr=0.15; theta=0.45; gain_cap=2.0; branch "
                       "prior = all movement channels; max_steps=200; 6 "
                       "episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0121.json"),
        "interpretation": ("BREAKS (0/4). No positive reward was ever "
                           "observed in any trajectory (the 0.99 goal tick "
                           "never occurred; every tick paid -0.01/-0.03), "
                           "so ZERO gain updates fired on all seeds: "
                           "learned is bit-identical to frozen (gains "
                           "exactly 1.0). Both arms unsolved (0/6 goals), "
                           "so the goal-count gate sits at floor -- the "
                           "mechanism finding (detector never fires) is "
                           "exact regardless. All ablations identical."),
        "limitations": ("4 seeds; task at floor for both arms (random "
                        "bandit cannot solve the maze in 6x200 ticks); "
                        "neutral specialists; branch prior given."),
    },
    "changing_rule": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on "
                       "changing_rule: R >= 1.30 on 4/4 fresh seeds"),
        "null": ("H0: R < 1.30 on >=1 seed -> ECR does not generalize to "
                 "changing_rule"),
        "metric": ("R = total(learned)/total(frozen) per seed, 12 "
                   "episodes x 40 steps; WIN iff R >= 1.30"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "conditions": ("changing_rule v1.0.0; neutral specialists; "
                       "capacity=4; gain_lr=0.15; theta=0.45; gain_cap=2.0; "
                       "branch_channels={a0,a1}; max_steps=40; 12 "
                       "episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0122.json"),
        "interpretation": ("BREAKS (0/4): R = 0.99/1.08/0.99/1.01. Dense "
                           "per-step +1.0 makes ECR's success path fire on "
                           "every correct tick; with no shaping stream the "
                           "progress-channel ID degenerates to "
                           "canonical-order tie-break (a0), and the "
                           "success lock-in pins a0 at the gain cap -- the "
                           "policy is ~always-a0, i.e. chance-level. All "
                           "three ablations are byte-identical to full "
                           "(no shaping ticks, all winners are branch "
                           "channels: no sub-mechanism ever triggers). "
                           "ECR's episodic machinery has no purchase on "
                           "dense-reward bandits."),
        "limitations": ("4 seeds; neutral specialists; PROTOCOL DEVIATION: "
                        "a timing pilot ran the single learned arm on seed "
                        "87301 before the battery; no code, parameter, or "
                        "gate changed as a result; the battery re-ran it "
                        "deterministically (G2 exact match 232.000000)."),
    },
    "compositional_rule": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on "
                       "compositional_rule: R >= 1.30 on 4/4 fresh seeds"),
        "null": ("H0: R < 1.30 on >=1 seed -> ECR does not generalize to "
                 "compositional_rule"),
        "metric": ("R = total(learned)/total(frozen) per seed, 12 "
                   "episodes x 40 steps; WIN iff R >= 1.30"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "limitations": "4 seeds; neutral specialists (bandit isolation).",
        "conditions": ("compositional_rule v1.0.0; neutral specialists; "
                       "capacity=4; gain_lr=0.15; theta=0.45; gain_cap=2.0; "
                       "branch_channels={a0,a1}; max_steps=40; 12 "
                       "episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0123.json"),
        "interpretation": ("BREAKS (0/4): R = 0.95/1.11/0.95/1.00. Same "
                           "degeneracy as changing_rule (dense +1.0, no "
                           "shaping -> a0 pinned at cap by the tie-break, "
                           "chance-level), PLUS the XOR contingency is "
                           "unrepresentable by the cue-blind bandit "
                           "regardless (single-cue marginals 50% by "
                           "construction). All ablations byte-identical "
                           "to full. ECR cannot learn contingencies that "
                           "require input conditioning (K10 bound "
                           "restated for ECR)."),
    },
    "cue_delayed_reward": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on "
                       "cue_delayed_reward: R >= 1.30 on 4/4 fresh seeds"),
        "null": ("H0: R < 1.30 on >=1 seed -> ECR does not generalize to "
                 "cue-conditioned contingencies"),
        "metric": ("R = total(learned)/total(frozen) per seed, 8 "
                   "episodes x <=15 steps; WIN iff R >= 1.30"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "conditions": ("cue_delayed_reward v1.0.0; neutral specialists; "
                       "capacity=4; gain_lr=0.15; theta=0.45; gain_cap=2.0; "
                       "branch_channels={branch_a,branch_b}; max_steps=15; "
                       "8 episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0124.json"),
        "interpretation": ("BREAKS by the 4/4 gate (3/4): R = 3.42 / 0.50 "
                           "/ 11.14 / 4.04. The correct branch varies per "
                           "episode with the cue; ECR's lock-in is "
                           "cue-blind: it helps when the locked branch "
                           "matches the episode's cue (3/4 seeds) and "
                           "cannot recover on mismatch (seed 87502: "
                           "frozen lucked 1 success, learned 0, R=0.50). "
                           "Failures emit no updates (r==0 post-branch), "
                           "so a wrong lock perseverates. Ablations: "
                           "no_prebranch 0/2 (pre-branch demotion "
                           "load-bearing -- t=0 poisoning returns without "
                           "it), no_boost 1/2, no_punish 1/2."),
        "limitations": ("4 seeds; neutral specialists; PROTOCOL DEVIATION: "
                        "a timing pilot ran the single learned arm on seed "
                        "87501 before the battery; no code, parameter, or "
                        "gate changed as a result; the battery re-ran it "
                        "deterministically (G2 exact match 0.820000)."),
    },
    "delayed_multistep": {
        "hypothesis": ("H1: ECR learned gains beat frozen gains on "
                       "delayed_multistep: R >= 1.30 on 4/4 fresh seeds"),
        "null": ("H0: R < 1.30 on >=1 seed -> ECR does not generalize to "
                 "multi-step conjunctions"),
        "metric": ("R = total(learned)/total(frozen) per seed, 8 "
                   "episodes x <=45 steps; WIN iff R >= 1.30"),
        "baseline": ("frozen = identical ReturnConditionedEpisodic"
                     "Arbitrator, frozen=True, gains pinned 1.0, paired "
                     "per seed"),
        "conditions": ("delayed_multistep v1.0.0; neutral specialists; "
                       "capacity=4; gain_lr=0.15; theta=0.45; gain_cap=2.0; "
                       "branch_channels={branch_a,branch_b}; max_steps=45; "
                       "8 episodes/arm; prereg experiments/"
                       "preregistration_EXP-FP-0125.json"),
        "interpretation": ("HOLDS (4/4): R = 3.15 / 2.01 / 1.68 / 2.46. "
                           "The terminal tick pays 0.02+1.00=1.02 >= 1.0, "
                           "so the success detector fires correctly. The "
                           "first-branch lock-in + forward corridor "
                           "machinery carry the win; the junction reuses "
                           "the locked branch gains, solving the "
                           "conjunction when pattern==(locked,locked). On "
                           "seed 87603 (0 successes) the shaping-only edge "
                           "still clears the gate (R=1.68): the corridor "
                           "machinery alone beats frozen's wandering. "
                           "Ablations: no_prebranch 0/2 (load-bearing), "
                           "no_boost 1/2, no_punish 2/2 (shaping demotion "
                           "not load-bearing here). Caveat: the lock is "
                           "one-shot (no re-choice mechanism); ECR-episodes "
                           "bleed across env episodes on failed runs "
                           "(ledger clears only on r>=1.0 or 45-tick "
                           "truncation) -- same property as EXP-FP-0080."),
        "limitations": ("4 seeds; neutral specialists; the win's "
                        "conjunction component depends on the first "
                        "success's branch matching later patterns "
                        "(luck-of-the-lock); not tested with adversarial "
                        "pattern schedules."),
    },
}


def main():
    receipts_dir = os.path.join(FLESH, "receipts")
    for env in ENVS:
        path = os.path.join(FLESH, "var", f"ecr-gen-{env}-results.json")
        with open(path) as f:
            result = json.load(f)
        t = TEXT[env]
        assert result["experiment_id"].startswith("EXP-FP-012"), \
            f"staged ID mismatch: {result['experiment_id']}"
        out = write_receipt(
            result, receipts_dir,
            hypothesis=t["hypothesis"], null=t["null"],
            preregistered_metric=t["metric"], baseline=t["baseline"],
            conditions=t["conditions"], interpretation=t["interpretation"],
            limitations=t["limitations"])
        print("receipt:", out)
    ok, problems = verify_chain(receipts_dir)
    print(f"verify_chain: {'OK' if ok else 'PROBLEMS: ' + str(problems)}")
    if not ok:
        # Report but do not fail the run: pre-existing historical chain
        # issues (EXP-AB-K3C, EXP-FP-0040, hashless legacy receipts) are
        # preserved per lab law; only OUR six receipts must chain cleanly.
        ours = [p for p in problems if "EXP-FP-012" in p]
        if ours:
            raise SystemExit(f"OUR receipts have chain problems: {ours}")
        print("chain problems are pre-existing/historical only; "
              "EXP-FP-012x receipts chain cleanly.")


if __name__ == "__main__":
    main()
