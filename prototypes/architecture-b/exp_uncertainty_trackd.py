"""Track D — metacognition hazard program: decision-usable uncertainty.

Four preregistered estimator families, one shared gate:

  EXP-FP-0110 (a) conformal prediction intervals — one-sided conformal
      transducer on a rolling fit/calibration split (clean-room stdlib
      reimplementation of the SplitConformalRegressor score-quantile math;
      MAPIE donor D17 is ADOPT but NOT vendored — H11 is PACKAGED ONLY).
  EXP-FP-0111 (b) empirical error-quantile tracking — Laplace-smoothed
      empirical event rate over a trailing window (no distributions).
  EXP-FP-0112 (c) ensemble disagreement — N=5 ArchB members, different
      predictor init seeds; p = member vote fraction.
  EXP-FP-0113 (d) phase-stratified climatology — the lane's own idea:
      empirical rate stratified by episode_idx mod 5 (the flip cycle).

Shared gate (program step 1): per-domain Brier_est < Brier_climo
(sequential Laplace climatology) on >=3/4 domains, scored by
benchmarks/calibration_battery.py score_domain on the same four domains
with the same frozen thresholds (EPS_OBS=0.2682, EPS_RW=0.2067,
TAU_FAIL=0.4156). corr(sigma,|err|) reported, no threshold.
REJECT on failure: full negative-result entry, no softening.

Stages:
  --run-reference   5 fresh seeds x 10 eps changing_rule (replicates
                    calibration_battery.run_seed exactly) -> JSONL logs
  --check-determinism  recompute seed[0] reference -> byte compare (G1)
  --run-ensemble    5 members x 5 seeds x 10 eps -> JSONL logs
  --score           run all four estimators + ablations over the logs,
                    print the gate table, write results JSON

Deterministic. Stdlib only (no random.* module calls; seeded RNG only via
env_interface.derive_seed where a stream is needed).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FP_DIR, "experiments"))
sys.path.insert(0, os.path.join(FP_DIR, "benchmarks"))

from env_interface import config_hash, derive_seed  # noqa: E402
from experiments import ENVS_BY_NAME, make_agent, run_closed_loop  # noqa: E402
from calibration_battery import score_domain  # noqa: E402  (the gate, verbatim)

# -- frozen protocol constants (CALIB-01) ------------------------------------
EPS_OBS = 0.2682
EPS_RW = 0.2067
TAU_FAIL = 0.4156
SEEDS = [73601, 73602, 73603, 73604, 73605]
N_EPISODES = 10
N_MEMBERS = 5
OUT_DIR = os.path.join(HERE, "experiments_out")
DOMAIN_ORDER = ["next_obs", "action_consequence",
                "competence_failure", "retrieval_usefulness"]


def _clamp01(p):
    return min(1.0 - 1e-9, max(1e-9, p))


# -- stage 1: reference + ensemble streams -----------------------------------

def run_reference(seed, out_dir, tag="EXP-FP-011X"):
    """Byte-faithful replicate of calibration_battery.run_seed."""
    logp = os.path.join(out_dir, f"{tag}.seed{seed}.predictions.jsonl")
    agent = make_agent(ENVS_BY_NAME["changing_rule"], seed=seed,
                       log_path=logp, affect="none")
    eps = run_closed_loop("changing_rule", agent, N_EPISODES, seed)
    if agent.plog:
        agent.plog.close()
    return logp, eps


def run_ensemble(seed, out_dir, tag="EXP-FP-0112", n_members=N_MEMBERS):
    """N independent ArchB members; distinct predictor init seeds."""
    logs = []
    for m in range(n_members):
        logp = os.path.join(
            out_dir, f"{tag}.seed{seed}.m{m}.predictions.jsonl")
        aseed = derive_seed(seed, m, "ensemble-member")
        agent = make_agent(ENVS_BY_NAME["changing_rule"], seed=aseed,
                           log_path=logp, affect="none")
        run_closed_loop("changing_rule", agent, N_EPISODES, seed)
        if agent.plog:
            agent.plog.close()
        logs.append(logp)
    return logs


# -- stream parsing -----------------------------------------------------------
# Per-domain target series in tick order: {"D1": {"y": [...], "ep": [...]}, ...}
# D1/D3: y = mean_abs_error (next_obs); D2: y = abs_error (action_conseq);
# D4: y = actual benefit (retrieval_usefulness). ep = episode index.

def parse_stream(log_path):
    series = {k: {"y": [], "ep": []} for k in ("D1", "D2", "D3", "D4")}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            t = r["target"]
            ep = r["episode"]
            if t == "next_obs":
                y = r["mean_abs_error"]
                series["D1"]["y"].append(y)
                series["D1"]["ep"].append(ep)
                series["D3"]["y"].append(y)
                series["D3"]["ep"].append(ep)
            elif t == "action_consequence":
                series["D2"]["y"].append(r["abs_error"])
                series["D2"]["ep"].append(ep)
            elif t == "retrieval_usefulness":
                series["D4"]["y"].append(r["actual"])
                series["D4"]["ep"].append(ep)
    return series


def pooled_series(log_paths):
    """Concatenate per-seed streams in seed order (CALIB-01 pooling)."""
    out = {k: {"y": [], "ep": []} for k in ("D1", "D2", "D3", "D4")}
    for lp in log_paths:
        s = parse_stream(lp)
        for k in out:
            out[k]["y"].extend(s[k]["y"])
            out[k]["ep"].extend(s[k]["ep"])
    return out


def events_of(domain, y):
    if domain == "D1":
        return [1.0 if v <= EPS_OBS else 0.0 for v in y]
    if domain == "D2":
        return [1.0 if v <= EPS_RW else 0.0 for v in y]
    if domain == "D3":
        return [1.0 if v > TAU_FAIL else 0.0 for v in y]
    if domain == "D4":
        return [1.0 if v > 0 else 0.0 for v in y]
    raise ValueError(domain)


# -- estimators (online, strictly causal) -------------------------------------

class _OnlineBase:
    def __init__(self):
        self._y = []

    def _push(self, y):
        self._y.append(float(y))

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    @staticmethod
    def _std(xs):
        n = len(xs)
        if n < 2:
            return 1.0
        m = sum(xs) / n
        return math.sqrt(sum((x - m) ** 2 for x in xs) / n) or 1.0


class ConformalTransducer(_OnlineBase):
    """(a) One-sided conformal transducer on rolling fit/calibration split.

    yhat_t = mean of last F targets; s_i = y_i - yhat_i over the last C
    scores; p = empirical score-CDF at the event boundary (Vovk's conformal
    predictive distribution, one-sided). Warmup t < F+C: p = 0.5 (scored).
    """

    F = 100
    C = 200

    def __init__(self):
        super().__init__()
        self._yhat_at = []

    def predict(self, domain):
        t = len(self._y)
        yhat = self._mean(self._y[max(0, t - self.F):t])
        if t < self.F + self.C:
            return 0.5, 1.0
        scores = [self._y[i] - self._yhat_at[i]
                  for i in range(t - self.C, t)]
        sigma = self._std(scores)
        if domain in ("D1", "D2"):
            thr = EPS_OBS if domain == "D1" else EPS_RW
            p = sum(1 for s in scores if s < thr - yhat) / self.C
        elif domain == "D3":
            p = sum(1 for s in scores if s >= TAU_FAIL - yhat) / self.C
        else:  # D4
            p = sum(1 for s in scores if s >= 0.0 - yhat) / self.C
        return _clamp01(p), sigma

    def observe(self, y):
        y = float(y)
        t = len(self._y)
        self._yhat_at.append(self._mean(self._y[max(0, t - self.F):t]))
        self._push(y)

    def step(self, domain, y):
        p, sigma = self.predict(domain)
        self.observe(y)
        return p, sigma


class RollingRate(_OnlineBase):
    """(b) Laplace-smoothed empirical event rate over trailing window W."""

    def __init__(self, W=200):
        super().__init__()
        self.W = W
        self._o = []

    def predict(self, domain):
        t = len(self._y)
        lo = max(0, t - self.W)
        window_o = self._o[lo:t]
        Wt = len(window_o)
        p = (1.0 + sum(window_o)) / (2.0 + Wt)
        sigma = self._std(self._y[lo:t])
        return _clamp01(p), sigma

    def observe(self, domain, y):
        y = float(y)
        self._o.append(events_of(domain, [y])[0])
        self._push(y)

    def step(self, domain, y):
        p, sigma = self.predict(domain)
        self.observe(domain, y)
        return p, sigma


class PhaseStratified(_OnlineBase):
    """(d) Empirical rate stratified by episode_idx mod 5 (the flip cycle).

    Fallback to the pooled rate when the stratum has < 20 past ticks.
    """

    STRATUM_MIN = 20

    def __init__(self):
        super().__init__()
        self._o = []
        self._ep = []

    def _p_for(self, domain, ep, mod):
        t = len(self._y)
        k = int(ep) % mod
        stratum_hits = sum(1 for i in range(t)
                           if self._ep[i] % mod == k and self._o[i] > 0.5)
        stratum_n = sum(1 for i in range(t) if self._ep[i] % mod == k)
        if stratum_n >= self.STRATUM_MIN:
            p = (1.0 + stratum_hits) / (2.0 + stratum_n)
        else:
            p = (1.0 + sum(self._o)) / (2.0 + t)
        sigma = self._std(self._y[max(0, t - 200):t])
        return _clamp01(p), sigma

    def predict(self, domain, ep):
        return self._p_for(domain, ep, 5)

    def observe(self, domain, y, ep):
        y = float(y)
        self._o.append(events_of(domain, [y])[0])
        self._ep.append(int(ep))
        self._push(y)

    def step(self, domain, y, ep):
        p, sigma = self.predict(domain, ep)
        self.observe(domain, y, ep)
        return p, sigma


class GaussianBridgeAblation(_OnlineBase):
    """(a) ablation: rolling point forecast + CALIB-01-style Gaussian bridge."""

    F = 100

    def step(self, domain, y):
        y = float(y)
        t = len(self._y)
        win = self._y[max(0, t - self.F):t]
        yhat = self._mean(win)
        sigma = self._std(win)
        if domain == "D1":
            p = math.erf(EPS_OBS / (sigma * math.sqrt(2.0)))
        elif domain == "D2":
            p = math.erf(EPS_RW / (sigma * math.sqrt(2.0)))
        elif domain == "D3":
            p = math.erfc(TAU_FAIL / (sigma * math.sqrt(2.0)))
        else:  # D4: Phi(yhat/sigma), no bridge on the mean
            p = 0.5 * (1.0 + math.erf(yhat / (sigma * math.sqrt(2.0))))
        self._push(y)
        return _clamp01(p), sigma


# -- family runners ------------------------------------------------------------

def run_family_offline(series, family):
    """series: pooled per-domain {y, ep}. Returns {domain: triples}."""
    triples = {}
    for dom in ("D1", "D2", "D3", "D4"):
        y = series[dom]["y"]
        ep = series[dom]["ep"]
        o = events_of(dom, y)
        est = {"conformal": ConformalTransducer,
               "rolling200": lambda: RollingRate(W=200),
               "rolling50": lambda: RollingRate(W=50),
               "stratified": PhaseStratified,
               "gauss_abl": GaussianBridgeAblation}[family]()
        tri = []
        for t, (yv, ov, ev) in enumerate(zip(y, o, ep)):
            if isinstance(est, PhaseStratified):
                p, sigma = est.step(dom, yv, ev)
            else:
                p, sigma = est.step(dom, yv)
            tri.append((p, ov, sigma, abs(yv)))
        triples[dom] = tri
    return triples


class PhaseStratifiedMod2(PhaseStratified):
    """(d) negative-control ablation: stratify by episode_idx mod 2."""

    def predict(self, domain, ep):  # noqa: D102
        return self._p_for(domain, ep, 2)


def run_stratified_mod2(series):
    """(d) negative-control ablation: stratify by episode_idx mod 2."""
    triples = {}
    for dom in ("D1", "D2", "D3", "D4"):
        y = series[dom]["y"]
        ep = series[dom]["ep"]
        o = events_of(dom, y)
        est = PhaseStratifiedMod2()
        tri = []
        for yv, ov, ev in zip(y, o, ep):
            p, sigma = est.step(dom, yv, ev)
            tri.append((p, ov, sigma, abs(yv)))
        triples[dom] = tri
    return triples


def run_ensemble_family(member_log_paths):
    """(c) p = member vote fraction; o = member 0's event."""
    per_member = [parse_stream(lp) for lp in member_log_paths]
    # truncate to min tick count per domain (preregistered)
    trunc = {}
    for dom in ("D1", "D2", "D3", "D4"):
        n = min(len(pm[dom]["y"]) for pm in per_member)
        trunc[dom] = n
    triples = {}
    for dom in ("D1", "D2", "D3", "D4"):
        n = trunc[dom]
        tri = []
        for t in range(n):
            ys = [pm[dom]["y"][t] for pm in per_member]
            os_ = [events_of(dom, [yv])[0] for yv in ys]
            p = sum(os_) / len(os_)
            sigma = (math.sqrt(sum((v - sum(ys) / len(ys)) ** 2
                                   for v in ys) / len(ys))
                     if len(ys) > 1 else 1.0) or 1.0
            tri.append((_clamp01(p), os_[0], sigma, abs(ys[0])))
        triples[dom] = tri
    return triples, trunc


def run_member0_gauss_ablation(member0_log):
    """(c) ablation: member 0 alone + CALIB-01 Gaussian bridge on its sigma."""
    tri = {"D1": [], "D2": [], "D3": [], "D4": []}
    with open(member0_log) as fh:
        for line in fh:
            r = json.loads(line)
            t = r["target"]
            s = r["uncertainty"]
            if t == "next_obs":
                e = r["mean_abs_error"]
                p1 = math.erf(EPS_OBS / (s * math.sqrt(2.0))) if s > 0 else 0.5
                p3 = (math.erfc(TAU_FAIL / (s * math.sqrt(2.0)))
                      if s > 0 else 0.5)
                tri["D1"].append((_clamp01(p1), 1.0 if e <= EPS_OBS else 0.0,
                                  s, e))
                tri["D3"].append((_clamp01(p3), 1.0 if e > TAU_FAIL else 0.0,
                                  s, e))
            elif t == "action_consequence":
                e = r["abs_error"]
                p2 = math.erf(EPS_RW / (s * math.sqrt(2.0))) if s > 0 else 0.5
                tri["D2"].append((_clamp01(p2), 1.0 if e <= EPS_RW else 0.0,
                                  s, e))
            elif t == "retrieval_usefulness":
                mu = r["prediction"]
                b = r["actual"]
                p4 = (0.5 * (1.0 + math.erf(mu / (s * math.sqrt(2.0))))
                      if s > 0 else 0.5)
                tri["D4"].append((_clamp01(p4), 1.0 if b > 0 else 0.0,
                                  s, abs(b)))
    return tri


NAMES = {"D1": "next_obs", "D2": "action_consequence",
         "D3": "competence_failure", "D4": "retrieval_usefulness"}


def gate_table(triples):
    """Score every domain with the CALIB-01 gate; return verdicts."""
    out = {}
    for dom in ("D1", "D2", "D3", "D4"):
        sc = score_domain(NAMES[dom], triples[dom])
        out[dom] = sc
    beats = sum(1 for dom in out
                if out[dom].get("brier_B") is not None
                and out[dom].get("brier_climo") is not None
                and out[dom]["brier_B"] < out[dom]["brier_climo"])
    out["_gate"] = {"domains_beaten": beats,
                    "verdict": "PASS" if beats >= 3 else "REJECT"}
    return out


# -- CLI ------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Track D uncertainty families")
    ap.add_argument("--stage", required=True,
                    choices=["run-reference", "check-determinism",
                             "run-ensemble", "score",
                             "run-shadow", "score-0114", "du1", "du2"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--family", default="stratified",
                    choices=["stratified", "ensemble"],
                    help="estimator family for the DU stages")
    ap.add_argument("--tag", default=None,
                    help="log tag override for DU stages")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.stage == "run-reference":
        logs = []
        for seed in SEEDS:
            print(f"--- reference seed {seed} ---", flush=True)
            logp, _ = run_reference(seed, args.out)
            logs.append(logp)
        print(json.dumps({"logs": logs}))

    elif args.stage == "check-determinism":
        import hashlib
        seed = SEEDS[0]
        logp, _ = run_reference(seed, args.out, tag="EXP-FP-011X-determinism")
        with open(logp, "rb") as f:
            h1 = hashlib.sha256(f.read()).hexdigest()
        ref = os.path.join(args.out, f"EXP-FP-011X.seed{seed}.predictions.jsonl")
        with open(ref, "rb") as f:
            h0 = hashlib.sha256(f.read()).hexdigest()
        ok = h0 == h1
        print(json.dumps({"seed": seed, "match": ok,
                          "hash": h0[:16]}))
        if not ok:
            raise SystemExit("G1 determinism spot-check FAILED")

    elif args.stage == "run-ensemble":
        all_logs = {}
        for seed in SEEDS:
            print(f"--- ensemble seed {seed} ---", flush=True)
            all_logs[str(seed)] = run_ensemble(seed, args.out)
        print(json.dumps({"member_logs": all_logs}))

    elif args.stage == "score":
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ref_logs = [os.path.join(args.out, f"EXP-FP-011X.seed{s}.predictions.jsonl")
                    for s in SEEDS]
        for lp in ref_logs:
            if not os.path.exists(lp):
                raise SystemExit(f"missing reference log {lp}: run --run-reference first")
        series = pooled_series(ref_logs)
        n_ticks = {d: len(series[d]["y"]) for d in series}
        print(f"pooled ticks per domain: {n_ticks}", flush=True)

        results = {"started_utc": started, "seeds": SEEDS,
                   "n_ticks_pooled": n_ticks, "families": {}}

        families = [
            ("EXP-FP-0110", "conformal"),
            ("EXP-FP-0110-abl", "gauss_abl"),
            ("EXP-FP-0111", "rolling200"),
            ("EXP-FP-0111-abl", "rolling50"),
            ("EXP-FP-0113", "stratified"),
            ("EXP-FP-0113-abl", "stratified_mod2"),
        ]
        for exp_id, fam in families:
            print(f"--- scoring {exp_id} ({fam}) ---", flush=True)
            if fam == "stratified_mod2":
                tri = run_stratified_mod2(series)
            else:
                tri = run_family_offline(series, fam)
            gt = gate_table(tri)
            results["families"][exp_id] = {
                "family": fam,
                "gate": gt["_gate"],
                "domains": {NAMES[d]: {k: gt[d].get(k) for k in
                                       ("n", "base_rate", "brier_B",
                                        "brier_climo", "brier_const05",
                                        "skill_vs_climo", "ece", "mce",
                                        "signed_error", "corr_sigma_err",
                                        "verdict")}
                            for d in ("D1", "D2", "D3", "D4")},
            }
            g = gt["_gate"]
            print(f"  gate: {g['verdict']} ({g['domains_beaten']}/4 domains beaten)")

        # (c) ensemble
        print("--- scoring EXP-FP-0112 (ensemble) ---", flush=True)
        ens_triples_all, trunc_report = {}, {}
        for seed in SEEDS:
            mlogs = [os.path.join(args.out, f"EXP-FP-0112.seed{seed}.m{m}.predictions.jsonl")
                     for m in range(N_MEMBERS)]
            for lp in mlogs:
                if not os.path.exists(lp):
                    raise SystemExit(f"missing ensemble log {lp}: run --run-ensemble first")
            tri, trunc = run_ensemble_family(mlogs)
            ens_triples_all[str(seed)] = tri
            trunc_report[str(seed)] = trunc
        pooled_tri = {d: [] for d in ("D1", "D2", "D3", "D4")}
        for seed in SEEDS:
            for d in pooled_tri:
                pooled_tri[d].extend(ens_triples_all[str(seed)][d])
        gt = gate_table(pooled_tri)
        results["families"]["EXP-FP-0112"] = {
            "family": "ensemble", "gate": gt["_gate"],
            "truncation": trunc_report,
            "domains": {NAMES[d]: {k: gt[d].get(k) for k in
                                   ("n", "base_rate", "brier_B", "brier_climo",
                                    "brier_const05", "skill_vs_climo", "ece",
                                    "mce", "signed_error", "corr_sigma_err",
                                    "verdict")}
                        for d in ("D1", "D2", "D3", "D4")},
        }
        print(f"  gate: {gt['_gate']['verdict']} ({gt['_gate']['domains_beaten']}/4)")

        # (c) ablation: member 0 + Gaussian bridge
        print("--- scoring EXP-FP-0112-abl (member0+gauss) ---", flush=True)
        abl_tri = {d: [] for d in ("D1", "D2", "D3", "D4")}
        for seed in SEEDS:
            m0 = os.path.join(args.out, f"EXP-FP-0112.seed{seed}.m0.predictions.jsonl")
            t0 = run_member0_gauss_ablation(m0)
            for d in abl_tri:
                abl_tri[d].extend(t0[d])
        gt = gate_table(abl_tri)
        results["families"]["EXP-FP-0112-abl"] = {
            "family": "member0_gauss_ablation", "gate": gt["_gate"],
            "domains": {NAMES[d]: {k: gt[d].get(k) for k in
                                   ("n", "base_rate", "brier_B", "brier_climo",
                                    "brier_const05", "skill_vs_climo", "ece",
                                    "mce", "signed_error", "corr_sigma_err",
                                    "verdict")}
                        for d in ("D1", "D2", "D3", "D4")},
        }
        print(f"  gate: {gt['_gate']['verdict']} ({gt['_gate']['domains_beaten']}/4)")

        results["written_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        outp = os.path.join(args.out, "trackd_gate_results.json")
        with open(outp, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"results -> {outp}")

    elif args.stage == "run-shadow":
        for seed in SEEDS:
            print(f"--- shadow seed {seed} ---", flush=True)
            run_shadow_ensemble(seed, args.out)
        print("shadow ensemble complete (G1 primary-consistency passed)")

    elif args.stage == "score-0114":
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pooled_tri = {"D1": [], "D2": [], "D3": [], "D4": []}
        for seed in SEEDS:
            print(f"--- scoring shadow seed {seed} ---", flush=True)
            tri = score_shadow_ensemble(seed, args.out)
            for d in pooled_tri:
                pooled_tri[d].extend(tri[d])
        gt = gate_table(pooled_tri)
        # ablation: B's stated sigma + CALIB-01 bridge on the reference logs
        ref_logs = [os.path.join(args.out, f"EXP-FP-011X.seed{s}.predictions.jsonl")
                    for s in SEEDS]
        abl_tri = score_stated_sigma_ablation(ref_logs)
        gt_abl = gate_table(abl_tri)
        results = {
            "experiment_id": "EXP-FP-0114",
            "started_utc": started,
            "written_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "seeds": SEEDS,
            "n_members": N_MEMBERS,
            "gate": gt["_gate"],
            "domains": {NAMES[d]: {k: gt[d].get(k) for k in
                                   ("n", "base_rate", "brier_B", "brier_climo",
                                    "brier_const05", "skill_vs_climo", "ece",
                                    "mce", "signed_error", "corr_sigma_err",
                                    "verdict")}
                        for d in ("D1", "D2", "D3", "D4")},
            "ablation_stated_sigma": {
                "gate": gt_abl["_gate"],
                "domains": {NAMES[d]: {k: gt_abl[d].get(k) for k in
                                       ("n", "base_rate", "brier_B",
                                        "brier_climo", "skill_vs_climo",
                                        "corr_sigma_err", "verdict")}
                            for d in ("D1", "D2", "D3", "D4")}},
        }
        print(f"EXP-FP-0114 gate: {gt['_gate']['verdict']} "
              f"({gt['_gate']['domains_beaten']}/4)")
        print(f"ablation (stated sigma): {gt_abl['_gate']['verdict']} "
              f"({gt_abl['_gate']['domains_beaten']}/4)")
        outp = os.path.join(args.out, "trackd_0114_results.json")
        with open(outp, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"results -> {outp}")

    elif args.stage in ("du1", "du2"):
        tag = args.tag or f"EXP-FP-DU-{args.stage.upper()}-{args.family}"
        runner = run_du1 if args.stage == "du1" else run_du2
        summary = {}
        for seed in DU_SEEDS:
            print(f"--- {args.stage} {args.family} seed {seed} ---", flush=True)
            g = runner(seed, args.out, tag, args.family, gated=True)
            b = runner(seed, args.out, tag, args.family, gated=False)
            gm = sum(g["returns"]) / len(g["returns"])
            bm = sum(b["returns"]) / len(b["returns"])
            summary[str(seed)] = {
                "gated_mean": gm, "blind_mean": bm, "delta": gm - bm,
                "gated": g, "blind": b,
            }
            print(f"  gated={gm:.3f} blind={bm:.3f} delta={gm-bm:+.3f} "
                  f"fire_rate={g.get('fire_rate', g.get('apply_rate')):.3f}")
        deltas = [v["delta"] for v in summary.values()]
        agree = sum(1 for d in deltas if d > 0)
        mean_delta = sum(deltas) / len(deltas)
        win = mean_delta > 0 and agree >= 3
        summary["_verdict"] = {"mean_delta": mean_delta,
                               "seeds_agree": agree,
                               "win": win,
                               "rule": ("WIN" if win else "NO-WIN") +
                               f": seed-mean delta {mean_delta:+.4f}, "
                               f"{agree}/4 seeds positive"}
        print(f"verdict: {summary['_verdict']['rule']}")
        outp = os.path.join(args.out, f"{tag}.summary.json")
        with open(outp, "w") as f:
            json.dump({"experiment": tag, "family": args.family,
                       "du": args.stage, "seeds": DU_SEEDS,
                       "written_utc": datetime.datetime.now(
                           datetime.timezone.utc).isoformat(),
                       "summary": summary}, f, indent=2, sort_keys=True)
        print(f"results -> {outp}")


# -- EXP-FP-0114: honest shadow-mode ensemble ------------------------------------
# Primary ArchB drives the env; 4 shadow members (distinct predictor init
# seeds) replay the primary's actions via replay_actions. All p_t derive
# from PRE-OUTCOME predictions only (no realized-error leakage).

def run_shadow_ensemble(seed, out_dir, tag="EXP-FP-0114"):
    """Primary + 4 shadows; returns (primary_log, shadow_logs).

    G1 consistency: the re-run primary JSONL must be byte-identical to the
    reference EXP-FP-011X log (shadows cannot perturb the primary).
    """
    import hashlib
    env_cls = ENVS_BY_NAME["changing_rule"]
    ref_log = os.path.join(out_dir, f"EXP-FP-011X.seed{seed}.predictions.jsonl")
    if not os.path.exists(ref_log):
        raise SystemExit(f"missing reference log {ref_log}")
    primary_log = os.path.join(
        out_dir, f"{tag}.seed{seed}.primary.predictions.jsonl")
    shadow_logs = [os.path.join(
        out_dir, f"{tag}.seed{seed}.shadow{m}.predictions.jsonl")
        for m in range(1, N_MEMBERS)]
    primary = make_agent(env_cls, seed=seed, log_path=primary_log,
                         affect="none")
    shadows = [make_agent(
        env_cls, seed=derive_seed(seed, m, "ensemble-member"),
        log_path=sl, affect="none")
        for m, sl in zip(range(1, N_MEMBERS), shadow_logs)]
    env = env_cls()
    for run_index in range(N_EPISODES):
        episode_seed = derive_seed(seed, run_index, "env")
        obs = env.reset(episode_seed)
        rseed = derive_seed(episode_seed, run_index, "agent")
        primary.reset(rseed, env.action_space())
        for sh in shadows:
            sh.reset(rseed, env.action_space())
        done = False
        while not done:
            a = primary.act(obs)
            for sh in shadows:
                sh.replay_actions = [a]
                sh.act(obs)
            obs2, reward, done, info = env.step(a)
            primary.update(obs, a, reward, done, info)
            for sh in shadows:
                sh.update(obs, a, reward, done, info)
            obs = obs2
    if primary.plog:
        primary.plog.close()
    for sh in shadows:
        if sh.plog:
            sh.plog.close()
    h_new = hashlib.sha256(open(primary_log, "rb").read()).hexdigest()
    h_ref = hashlib.sha256(open(ref_log, "rb").read()).hexdigest()
    if h_new != h_ref:
        raise SystemExit(
            f"G1 FAILED for seed {seed}: re-run primary differs from reference")
    return primary_log, shadow_logs


def _parse_predictions(log_path):
    """Per-tick pre-outcome predictions + outcomes, in tick order."""
    recs = {"next_obs": [], "action_consequence": [],
            "retrieval_usefulness": []}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["target"]].append(r)
    return recs


def _member_std(vals):
    n = len(vals)
    if n < 2:
        return 1.0
    if isinstance(vals[0], (list, tuple)):
        d = len(vals[0])
        per_dim = []
        for j in range(d):
            col = [v[j] for v in vals]
            m = sum(col) / n
            per_dim.append(math.sqrt(sum((x - m) ** 2 for x in col) / n))
        return sum(per_dim) / d or 1.0
    m = sum(vals) / n
    return math.sqrt(sum((x - m) ** 2 for x in vals) / n) or 1.0


def _bridge_p(domain, sigma, uhat_bar=None):
    sigma = max(sigma, 1e-9)
    if domain in ("D1", "D2"):
        thr = EPS_OBS if domain == "D1" else EPS_RW
        return _clamp01(math.erf(thr / (sigma * math.sqrt(2.0))))
    if domain == "D3":
        return _clamp01(math.erfc(TAU_FAIL / (sigma * math.sqrt(2.0))))
    return _clamp01(0.5 * (1.0 + math.erf(uhat_bar / (sigma * math.sqrt(2.0)))))


def score_shadow_ensemble(seed, out_dir, tag="EXP-FP-0114"):
    """Honest (c): p from pre-outcome member predictions; o = primary's."""
    primary_log = os.path.join(
        out_dir, f"{tag}.seed{seed}.primary.predictions.jsonl")
    shadow_logs = [os.path.join(
        out_dir, f"{tag}.seed{seed}.shadow{m}.predictions.jsonl")
        for m in range(1, N_MEMBERS)]
    logs = [primary_log] + shadow_logs
    parsed = [_parse_predictions(lp) for lp in logs]
    n_ticks = [len(p["next_obs"]) for p in parsed]
    if len(set(n_ticks)) != 1:
        raise SystemExit(f"tick-count mismatch across members: {n_ticks}")
    n = n_ticks[0]
    triples = {"D1": [], "D2": [], "D3": [], "D4": []}
    for t in range(n):
        # primary outcomes
        r0 = parsed[0]
        e1 = r0["next_obs"][t]["mean_abs_error"]
        e2 = r0["action_consequence"][t]["abs_error"]
        b4 = r0["retrieval_usefulness"][t]["actual"]
        o = {"D1": 1.0 if e1 <= EPS_OBS else 0.0,
             "D2": 1.0 if e2 <= EPS_RW else 0.0,
             "D3": 1.0 if e1 > TAU_FAIL else 0.0,
             "D4": 1.0 if b4 > 0 else 0.0}
        # member predictions (pre-outcome)
        xhats = [p["next_obs"][t]["prediction"] for p in parsed]
        rhats = [p["action_consequence"][t]["prediction"] for p in parsed]
        uhats = [p["retrieval_usefulness"][t]["prediction"] for p in parsed]
        sig_obs = _member_std(xhats)
        sig_r = _member_std(rhats)
        sig_u = _member_std(uhats)
        uhat_bar = sum(uhats) / len(uhats)
        sig = {"D1": sig_obs, "D2": sig_r, "D3": sig_obs, "D4": sig_u}
        for dom in ("D1", "D2", "D3", "D4"):
            p = _bridge_p(dom, sig[dom],
                          uhat_bar if dom == "D4" else None)
            err = {"D1": e1, "D2": e2, "D3": e1, "D4": abs(b4)}[dom]
            triples[dom].append((p, o[dom], sig[dom], err))
    return triples


def score_stated_sigma_ablation(ref_logs):
    """(c) ablation: B's own stated sigma + CALIB-01 bridge, fresh seeds."""
    from calibration_battery import collect as calib_collect
    pooled = {"D1": [], "D2": [], "D3": [], "D4": []}
    for lp in ref_logs:
        c = calib_collect(lp)
        for k in pooled:
            pooled[k].extend(c[k])
    return pooled


# -- online machinery for the decision-use tests --------------------------------

class JSONLTailer:
    """Read newly appended JSONL lines (the §30 stream) causally."""

    def __init__(self, path):
        self._fh = open(path)
        self._pos = 0

    def read_new(self):
        self._fh.seek(self._pos)
        lines = self._fh.readlines()
        self._pos = self._fh.tell()
        return [json.loads(l) for l in lines if l.strip()]

    def close(self):
        self._fh.close()


def feed_estimator(est, rec):
    """Feed one §30 record into a per-domain online estimator."""
    t = rec["target"]
    ep = rec["episode"]
    if t == "next_obs":
        y = rec["mean_abs_error"]
        est["D1"].observe("D1", y, ep)
        est["D3"].observe("D3", y, ep)
    elif t == "action_consequence":
        est["D2"].observe("D2", rec["abs_error"], ep)
    elif t == "retrieval_usefulness":
        est["D4"].observe("D4", rec["actual"], ep)


DU_SEEDS = [73701, 73702, 73703, 73704]


def run_du1(seed, out_dir, tag, family, gated):
    """DU1: defer to random action when online p_D3 > 0.5 (tau frozen).

    family='stratified': single agent + JSONL tailer.
    family='ensemble': primary + 4 shadows; p_D3 from lagged shadow
      predictions (causality: the action must be fixed before
      action-conditional predictions exist; the lag is disclosed).
    """
    from env_interface import new_rng
    env_cls = ENVS_BY_NAME["changing_rule"]
    arm = "gated" if gated else "blind"
    logp = os.path.join(out_dir, f"{tag}.seed{seed}.{arm}.predictions.jsonl")
    defer_rng = new_rng(derive_seed(seed, 0, "defer"))
    env = env_cls()
    fired = 0
    ticks = 0
    returns = []

    if family == "stratified":
        agent = make_agent(env_cls, seed=seed, log_path=logp, affect="none")
        est = {"D3": PhaseStratified()}
        tailer = JSONLTailer(logp)
        lag_p = 0.5
        for run_index in range(N_EPISODES):
            episode_seed = derive_seed(seed, run_index, "env")
            obs = env.reset(episode_seed)
            agent.reset(derive_seed(episode_seed, run_index, "agent"),
                        env.action_space())
            done = False
            total = 0.0
            while not done:
                for rec in tailer.read_new():
                    if rec["target"] == "next_obs":
                        est["D3"].observe("D3", rec["mean_abs_error"],
                                          rec["episode"])
                if gated:
                    p_fail, _ = est["D3"].predict("D3", run_index)
                    if p_fail > 0.5:
                        agent.replay_actions = [defer_rng.randrange(2)]
                        fired += 1
                a = agent.act(obs)
                agent.plog._fh.flush()
                obs2, reward, done, info = env.step(a)
                agent.update(obs, a, reward, done, info)
                obs = obs2
                total += reward
                ticks += 1
            returns.append(total)
        tailer.close()
        if agent.plog:
            agent.plog.close()
    elif family == "ensemble":
        primary = make_agent(env_cls, seed=seed, log_path=logp, affect="none")
        shadows = [make_agent(
            env_cls, seed=derive_seed(seed, m, "ensemble-member"),
            log_path=None, affect="none") for m in range(1, N_MEMBERS)]
        lag_p = 0.5  # one-tick-lagged ensemble p_D3 (disclosed)
        for run_index in range(N_EPISODES):
            episode_seed = derive_seed(seed, run_index, "env")
            obs = env.reset(episode_seed)
            rseed = derive_seed(episode_seed, run_index, "agent")
            primary.reset(rseed, env.action_space())
            for sh in shadows:
                sh.reset(rseed, env.action_space())
            done = False
            total = 0.0
            while not done:
                if gated and lag_p > 0.5:
                    primary.replay_actions = [defer_rng.randrange(2)]
                    fired += 1
                a = primary.act(obs)
                members = [primary] + shadows
                for sh in shadows:
                    sh.replay_actions = [a]
                    sh.act(obs)
                # lagged p for NEXT tick from this tick's pre-outcome preds
                xhats = [m._pending["xhat"] for m in members]
                sig = _member_std(xhats)
                lag_p = _bridge_p("D3", sig)
                obs2, reward, done, info = env.step(a)
                primary.update(obs, a, reward, done, info)
                for sh in shadows:
                    sh.update(obs, a, reward, done, info)
                obs = obs2
                total += reward
                ticks += 1
            returns.append(total)
        if primary.plog:
            primary.plog.close()
    else:
        raise ValueError(family)
    return {"returns": returns, "fired": fired, "ticks": ticks,
            "fire_rate": fired / ticks if ticks else 0.0}


def run_du2(seed, out_dir, tag, family, gated):
    """DU2: gate the retrieval correction on online p_D4 > 0.5 (tau frozen).

    Uses the RetrievalGate 'p_ext' policy (reads driver-supplied p each tick).
    family='stratified': single agent + JSONL tailer.
    family='ensemble': primary + 4 shadows; p_D4 from lagged shadow uhats.
    """
    env_cls = ENVS_BY_NAME["changing_rule"]
    arm = "gated" if gated else "blind"
    logp = os.path.join(out_dir, f"{tag}.seed{seed}.{arm}.predictions.jsonl")
    env = env_cls()
    fired = 0
    avail = 0
    ticks = 0
    returns = []
    policy = "p_ext" if gated else "ungated"

    if family == "stratified":
        agent = make_agent(env_cls, seed=seed, log_path=logp, affect="none",
                           gate_policy=policy, gate_threshold=0.5)
        est = {"D4": PhaseStratified()}
        tailer = JSONLTailer(logp)
        for run_index in range(N_EPISODES):
            episode_seed = derive_seed(seed, run_index, "env")
            obs = env.reset(episode_seed)
            agent.reset(derive_seed(episode_seed, run_index, "agent"),
                        env.action_space())
            done = False
            total = 0.0
            while not done:
                for rec in tailer.read_new():
                    if rec["target"] == "retrieval_usefulness":
                        est["D4"].observe("D4", rec["actual"], rec["episode"])
                if gated:
                    p_use, _ = est["D4"].predict("D4", run_index)
                    agent.gate._p_ext = p_use
                a = agent.act(obs)
                agent.plog._fh.flush()
                obs2, reward, done, info = env.step(a)
                agent.update(obs, a, reward, done, info)
                obs = obs2
                total += reward
                ticks += 1
            returns.append(total)
        fired = agent.gate.n_applied
        avail = agent.gate.n_available
        tailer.close()
        if agent.plog:
            agent.plog.close()
    elif family == "ensemble":
        primary = make_agent(env_cls, seed=seed, log_path=logp, affect="none",
                             gate_policy=policy, gate_threshold=0.5)
        shadows = [make_agent(
            env_cls, seed=derive_seed(seed, m, "ensemble-member"),
            log_path=None, affect="none") for m in range(1, N_MEMBERS)]
        lag_p = 0.5
        for run_index in range(N_EPISODES):
            episode_seed = derive_seed(seed, run_index, "env")
            obs = env.reset(episode_seed)
            rseed = derive_seed(episode_seed, run_index, "agent")
            primary.reset(rseed, env.action_space())
            for sh in shadows:
                sh.reset(rseed, env.action_space())
            done = False
            total = 0.0
            while not done:
                if gated:
                    primary.gate._p_ext = lag_p
                a = primary.act(obs)
                members = [primary] + shadows
                for sh in shadows:
                    sh.replay_actions = [a]
                    sh.act(obs)
                uhats = [m._pending["uhat"] for m in members]
                sig = _member_std(uhats)
                ubar = sum(uhats) / len(uhats)
                lag_p = _bridge_p("D4", sig, ubar)
                obs2, reward, done, info = env.step(a)
                primary.update(obs, a, reward, done, info)
                for sh in shadows:
                    sh.update(obs, a, reward, done, info)
                obs = obs2
                total += reward
                ticks += 1
            returns.append(total)
        fired = primary.gate.n_applied
        avail = primary.gate.n_available
        if primary.plog:
            primary.plog.close()
    else:
        raise ValueError(family)
    return {"returns": returns, "fired": fired, "available": avail,
            "ticks": ticks,
            "apply_rate": fired / avail if avail else 0.0}


if __name__ == "__main__":
    main()
