"""Write Track D receipts (EXP-FP-0110/0111/0112/0113/0114) via harness.write_receipt.

Reads the gate/DU result JSONs + preregistrations; fails closed on existing
paths (never overwrite). Run once.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(FP, "experiments"))
sys.path.insert(0, os.path.join(FP, "prototypes", "architecture-b"))

from harness import write_receipt, verify_chain  # noqa: E402
from env_interface import config_hash  # noqa: E402

RECEIPTS_DIR = os.path.join(FP, "receipts")
OUT = os.path.join(FP, "prototypes", "architecture-b", "experiments_out")
EXP = os.path.join(FP, "experiments")

NAMES = {"D1": "next_obs", "D2": "action_consequence",
         "D3": "competence_failure", "D4": "retrieval_usefulness"}


def load_prereg(eid):
    with open(os.path.join(EXP, f"preregistration_{eid}.json")) as f:
        return json.load(f)


def domain_lines(domains):
    lines = []
    for dk in ("D1", "D2", "D3", "D4"):
        s = domains[NAMES[dk]]
        lines.append(
            f"{NAMES[dk]}: n={s.get('n')} base={s.get('base_rate')} "
            f"brier_B={s.get('brier_B')} brier_climo={s.get('brier_climo')} "
            f"skill={s.get('skill_vs_climo')} corr={s.get('corr_sigma_err')} "
            f"S={s.get('signed_error')} -> {s.get('verdict')}")
    return lines


def base_result(eid, prereg, started_utc, extra_config):
    cfg = {"experiment_id": eid,
           "track": "D (metacognition hazard program)",
           "protocol": "1950-tick CALIB-01 protocol (5 fresh seeds x 10 eps, "
                       "changing_rule v1.0.0, arch_b, affect=none)",
           "seeds": prereg["seeds"]}
    cfg.update(extra_config)
    return {"experiment_id": eid,
            "config": cfg,
            "config_hash": config_hash(cfg),
            "primary_seed": prereg["seeds"],
            "started_utc": started_utc,
            "episodes": {"note": "tick-level protocol; no episode arms",
                         "seeds": prereg["seeds"]},
            "summary": {}}


def main():
    gate = json.load(open(os.path.join(OUT, "trackd_gate_results.json")))
    r0114 = json.load(open(os.path.join(OUT, "trackd_0114_results.json")))
    du1 = json.load(open(os.path.join(OUT, "EXP-FP-DU1-0113.summary.json")))
    du2 = json.load(open(os.path.join(OUT, "EXP-FP-DU2-0113.summary.json")))

    # -- EXP-FP-0110 (a) conformal: REJECT ---------------------------------
    eid = "EXP-FP-0110"
    pr = load_prereg(eid)
    fam = gate["families"][eid]
    res = base_result(eid, pr, gate["started_utc"], {"family": "(a) conformal"})
    res["summary"] = {"gate": fam["gate"],
                      "domains": fam["domains"],
                      "ablation_gauss_bridge": gate["families"][
                          "EXP-FP-0110-abl"]["gate"]}
    res["episodes"]["n_ticks_pooled"] = gate["n_ticks_pooled"]
    interp = ("REJECTED. Conformal transducer (one-sided, rolling F=100/C=200) "
              "beats sequential climatology on 0/4 domains (skills -0.005..-0.272). "
              "The ablation (rolling point forecast + Gaussian bridge) also fails "
              "0/4. On a near-stationary stream the running average is already "
              "near-optimal; the conformal machinery adds estimation noise without "
              "adaptivity gain. Synthetic check confirms the transducer is "
              "well-calibrated on iid data (no implementation bug) — it is simply "
              "not better than climatology here. Full NR entry.")
    write_receipt(res, RECEIPTS_DIR, hypothesis=pr["hypothesis"], null=pr["null"],
                  preregistered_metric=pr["preregistered_metric"],
                  baseline=pr["baseline"], conditions=pr["conditions"],
                  interpretation=interp,
                  limitations="; ".join(pr["limitations_preregistered"]))
    print(f"wrote {eid}")

    # -- EXP-FP-0111 (b) rolling rate: REJECT --------------------------------
    eid = "EXP-FP-0111"
    pr = load_prereg(eid)
    fam = gate["families"][eid]
    res = base_result(eid, pr, gate["started_utc"], {"family": "(b) rolling"})
    res["summary"] = {"gate": fam["gate"],
                      "domains": fam["domains"],
                      "ablation_W50": gate["families"]["EXP-FP-0111-abl"]["gate"]}
    res["episodes"]["n_ticks_pooled"] = gate["n_ticks_pooled"]
    interp = ("REJECTED. Rolling-window (W=200) empirical rate beats climatology "
              "on 2/4 domains (D2 +0.002, D4 +0.003; D1/D3 fail) — below the "
              "preregistered >=3/4 bar. The W=50 ablation beats on 1/4 (D4 "
              "CALIBRATED). Recency carries almost no information beyond the "
              "running average on this stream: the event rates are too stable "
              "for windowing to help. Full NR entry.")
    write_receipt(res, RECEIPTS_DIR, hypothesis=pr["hypothesis"], null=pr["null"],
                  preregistered_metric=pr["preregistered_metric"],
                  baseline=pr["baseline"], conditions=pr["conditions"],
                  interpretation=interp,
                  limitations="; ".join(pr["limitations_preregistered"]))
    print(f"wrote {eid}")

    # -- EXP-FP-0112 (c) ensemble as-run: VOID --------------------------------
    eid = "EXP-FP-0112"
    pr = load_prereg(eid)
    fam = gate["families"][eid]
    res = base_result(eid, pr, gate["started_utc"],
                      {"family": "(c) ensemble, as-run (VOID)"})
    res["summary"] = {"gate": fam["gate"],
                      "domains": fam["domains"],
                      "status": "VOID — instrument invalid (see interpretation)",
                      "note": "Numbers below are NOT interpreted for any claim."}
    res["episodes"]["n_ticks_pooled"] = {"D1": 1950, "D2": 1950,
                                         "D3": 1950, "D4": 1950}
    interp = ("VOID — INSTRUMENT INVALID, no claim. The preregistered estimator "
              "computed p_t from members' REALIZED errors at tick t "
              "(vote fraction over contemporaneous outcomes). Sibling members' "
              "outcomes are correlated with the primary's (same env dynamics), "
              "so p_t peeked at the answer: contemporaneous-outcome leakage. "
              "The 4/4 'PASS' (skills +0.36..+0.49) is an artifact of the leak, "
              "not a property of ensemble disagreement. Caught at interpretation "
              "before any claim was absorbed. Redesignated as EXP-FP-0114 "
              "(shadow-mode, pre-outcome predictions only). "
              "Methodological negative result: a preregistered design can still "
              "be invalid; interpretation-time instrument audit is load-bearing.")
    write_receipt(res, RECEIPTS_DIR, hypothesis=pr["hypothesis"], null=pr["null"],
                  preregistered_metric=pr["preregistered_metric"],
                  baseline=pr["baseline"], conditions=pr["conditions"],
                  interpretation=interp,
                  limitations="; ".join(pr["limitations_preregistered"]))
    print(f"wrote {eid} (VOID)")

    # -- EXP-FP-0113 (d) stratified: PASS gate, NO-WIN DU ----------------------
    eid = "EXP-FP-0113"
    pr = load_prereg(eid)
    fam = gate["families"][eid]
    res = base_result(eid, pr, gate["started_utc"],
                      {"family": "(d) phase-stratified"})
    res["summary"] = {
        "gate": fam["gate"], "domains": fam["domains"],
        "ablation_mod2": gate["families"]["EXP-FP-0113-abl"]["gate"],
        "decision_use": {
            "DU1_defer": du1["summary"]["_verdict"],
            "DU1_detail": {k: v for k, v in du1["summary"].items()
                           if k != "_verdict"},
            "DU2_retrieval_gate": du2["summary"]["_verdict"],
            "DU2_detail": {k: v for k, v in du2["summary"].items()
                           if k != "_verdict"},
        }}
    res["episodes"]["n_ticks_pooled"] = gate["n_ticks_pooled"]
    interp = ("GATE PASS (3/4: D1 +0.003, D2 +0.005, D4 +0.019; D3 fails). "
              "Margins are thin; the mod-2 nonsense-stratification ablation "
              "fails 0/4, supporting that the flip-cycle structure (not "
              "stratification per se) is the active ingredient. "
              "DECISION-USE: DU1 (defer on p_D3>0.5) NO-WIN — fires on <1.5% of "
              "ticks (0.013/0.000/0.000/0.003), seed-mean delta -1.425, 0/4 "
              "positive; the rare deferrals hurt. DU2 (retrieval gate on "
              "p_D4>0.5) NO-WIN — apply rate 0.000 on all 4 seeds: cold-start "
              "collapse (p_init=0.5 not > 0.5 -> block -> realized benefit 0 -> "
              "p->0 -> never apply), the EXP-FP-0006 degeneracy recurring with "
              "an external p. VERDICT: calibrated-ish but decision-inert. "
              "Per the program rule, INTEGRATED at best — NOT decision-useful. "
              "No retrofit (no shadow/warm-start fix smuggled in post hoc).")
    write_receipt(res, RECEIPTS_DIR, hypothesis=pr["hypothesis"], null=pr["null"],
                  preregistered_metric=pr["preregistered_metric"],
                  baseline=pr["baseline"], conditions=pr["conditions"],
                  interpretation=interp,
                  limitations="; ".join(pr["limitations_preregistered"]))
    print(f"wrote {eid}")

    # -- EXP-FP-0114 (c) honest ensemble: REJECT --------------------------------
    eid = "EXP-FP-0114"
    pr = load_prereg(eid)
    res = base_result(eid, pr, r0114["started_utc"],
                      {"family": "(c) ensemble, shadow-mode (honest)",
                       "n_members": 5})
    res["summary"] = {
        "gate": r0114["gate"], "domains": r0114["domains"],
        "ablation_stated_sigma": r0114["ablation_stated_sigma"],
        "mechanism": ("median sigma_ens D1/D3 ~0.011 vs error scale ~0.27 "
                      "(members agree ~25x too closely); p saturates at "
                      "1.0/0.0 -> D1/D2 massively over-confident, D3 "
                      "UNMEASURABLE (degenerate p). corr(sigma_ens,|err|) D1 "
                      "= 0.29: ranking signal present, scale catastrophically "
                      "wrong for the Gaussian bridge.")}
    res["episodes"]["n_ticks_pooled"] = {"D1": 1950, "D2": 1950,
                                         "D3": 1950, "D4": 1950}
    interp = ("REJECTED 0/4 (D3 UNMEASURABLE counts as not-beaten). Honest "
              "pre-outcome ensemble disagreement over init seeds does NOT beat "
              "climatology: members converge to near-identical predictions "
              "(init-seed diversity washes out), so disagreement is ~25x too "
              "small for the Gaussian bridge and p saturates. The residual "
              "disagreement does couple to error magnitude (corr 0.29 on D1, "
              "the strongest coupling seen in this program) but the scale is "
              "wrong — a scale-calibrated mapping is a future experiment, not "
              "a retrofit. The ablation (B's stated sigma on the same fresh "
              "seeds) replicates CALIB-01: 0/4, corr 0.04-0.18. Full NR entry.")
    write_receipt(res, RECEIPTS_DIR, hypothesis=pr["hypothesis"], null=pr["null"],
                  preregistered_metric=pr["preregistered_metric"],
                  baseline=pr["baseline"], conditions=pr["conditions"],
                  interpretation=interp,
                  limitations="; ".join(pr["limitations_preregistered"]))
    print(f"wrote {eid}")

    print("chain:", verify_chain(RECEIPTS_DIR))


if __name__ == "__main__":
    main()
