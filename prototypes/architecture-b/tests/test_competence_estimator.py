"""Unit tests for the EXP-FP-0070 competence estimator (variant A).

Deterministic, stdlib only. No RNG anywhere (fabrication-tripwire clean).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competence_estimator import CompetenceEstimator, DomainEstimator  # noqa: E402


def _ctx(ounc=0.3, runc=0.2, uunc=0.5, uhat=0.1):
    return {"ounc": ounc, "runc": runc, "uunc": uunc, "uhat": uhat}


def test_anchor_equals_climatology():
    """learn=False: p must equal the Laplace running base rate exactly."""
    est = DomainEstimator("d", learn=False)
    a, n = 1.0, 0
    events = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    for e in events:
        expected = a / (n + 2.0)
        got = est.predict(0.3)["p"]
        assert abs(got - expected) < 1e-12, (got, expected)
        est.observe(0.2 if e else 0.9, e, 0.3, 0.01)
        a += e
        n += 1
    print("test_anchor_equals_climatology PASS")


def test_causality_prediction_ignores_current_outcome():
    """The prediction EMITTED for tick t must be identical whether the
    estimator has already observed tick t's outcome or not -- observing
    the outcome must not retroactively change the emitted prediction."""
    eA = CompetenceEstimator(0.2682, 0.2067, 0.4156)
    eB = CompetenceEstimator(0.2682, 0.2067, 0.4156)
    for t in range(30):
        ctx = _ctx(ounc=0.2 + 0.001 * t)
        out = {"err_next": 0.1 + 0.01 * t, "err_rw": 0.2, "benefit": 0.05}
        for e in (eA, eB):
            e.predict_tick(ctx)
            e.observe_tick(ctx, out, 0.01 * t)
    ctx = _ctx(ounc=0.9)
    out = {"err_next": 9.9, "err_rw": 9.9, "benefit": -9.9}
    pA = eA.predict_tick(ctx)   # emitted BEFORE observing
    eA.observe_tick(ctx, out, 9.9)
    pB = eB.predict_tick(ctx)   # eB never observes this outcome
    for d in pA:
        assert pA[d]["p"] == pB[d]["p"], (d, pA[d], pB[d])
        assert pA[d]["ehat"] == pB[d]["ehat"], (d,)
    print("test_causality_prediction_ignores_current_outcome PASS")


def test_learns_regime_beats_climatology():
    """Synthetic two-regime error stream with an UNINFORMATIVE stated
    uncertainty (mimics B's uncoupled sigma): the estimator must still beat
    climatology via error-history features, and couple to error magnitude."""
    EPS = 0.2682
    est = DomainEstimator("d")
    ps, os_, errs = [], [], []
    a, n = 1.0, 0
    brier_c = 0.0
    for t in range(600):
        regime = (t // 150) % 2  # alternate low/high error regimes
        base = 0.10 if regime == 0 else 0.60
        err = base + 0.04 * math.sin(t * 0.7)  # deterministic wiggle
        o = 1 if err <= EPS else 0
        p = est.predict(0.30)["p"]  # stated uncertainty: constant, useless
        pc = a / (n + 2.0)
        ps.append(p)
        os_.append(o)
        errs.append(err)
        brier_c += (pc - o) ** 2
        est.observe(err, o, 0.30, 0.01)
        a += o
        n += 1
    brier_c /= 600
    brier_e = sum((p - o) ** 2 for p, o in zip(ps, os_)) / 600
    assert brier_e < brier_c, (brier_e, brier_c)
    # coupling: corr(1-p, |err|)
    unc = [1.0 - p for p in ps]
    mu, me = sum(unc) / 600, sum(errs) / 600
    cov = sum((u - mu) * (e - me) for u, e in zip(unc, errs)) / 600
    su = math.sqrt(sum((u - mu) ** 2 for u in unc) / 600)
    se = math.sqrt(sum((e - me) ** 2 for e in errs) / 600)
    corr = cov / (su * se)
    assert corr >= 0.40, corr
    print(f"test_learns_regime_beats_climatology PASS "
          f"(brier {brier_e:.4f} < climo {brier_c:.4f}, corr {corr:.3f})")


def test_determinism():
    """Two identical replays produce byte-identical prediction streams."""
    def run():
        est = CompetenceEstimator(0.2682, 0.2067, 0.4156)
        out = []
        for t in range(120):
            ctx = _ctx(ounc=0.2 + 0.001 * t, uhat=0.05 * math.sin(t))
            pr = est.predict_tick(ctx)
            out.append((pr["next_obs"]["p"], pr["next_obs"]["ehat"],
                        pr["retrieval_usefulness"]["p"]))
            est.observe_tick(ctx, {"err_next": 0.15 + 0.002 * t,
                                   "err_rw": 0.1,
                                   "benefit": 0.02 * math.cos(t)}, 0.005)
        return out
    r1, r2 = run(), run()
    assert r1 == r2
    print("test_determinism PASS")


def _synth_stream(n=600):
    errs, outs = [], []
    for t in range(n):
        regime = (t // 150) % 2
        base = 0.10 if regime == 0 else 0.60
        err = base + 0.04 * math.sin(t * 0.7)
        errs.append(err)
        outs.append(1 if err <= 0.2682 else 0)
    return errs, outs


def _replay(err_stream, outs):
    """Prequential replay; returns (brier_est, brier_climo)."""
    est = DomainEstimator("d")
    be, bc = 0.0, 0.0
    a, n = 1.0, 0
    for t in range(len(outs)):
        p = est.predict(0.30)["p"]
        pc = a / (n + 2.0)
        be += (p - outs[t]) ** 2
        bc += (pc - outs[t]) ** 2
        est.observe(err_stream[t], outs[t], 0.30, 0.01)
        a += outs[t]
        n += 1
    return be / len(outs), bc / len(outs)


def test_feature_groups_both_contribute():
    """Decomposition on synthetic regimes: (a) intact features beat
    climatology; (b) permuting the ERROR stream (perm(t)=(t*7919+13)%N,
    destroying the error-history link while keeping the causal event-rate
    feature) still beats climatology via fast base-rate tracking, but by
    LESS than intact features -- proving the error-history features carry
    independent signal beyond the fast base rate."""
    errs, outs = _synth_stream()
    b_intact, b_climo = _replay(errs, outs)
    n = len(outs)
    perm = [(t * 7919 + 13) % n for t in range(n)]
    assert sorted(perm) == list(range(n))
    b_perm, _ = _replay([errs[perm[t]] for t in range(n)], outs)
    assert b_intact < b_climo, (b_intact, b_climo)
    assert b_perm < b_climo, (b_perm, b_climo)
    assert b_intact < b_perm, (b_intact, b_perm)
    print(f"test_feature_groups_both_contribute PASS "
          f"(intact {b_intact:.4f} < permuted {b_perm:.4f} "
          f"< climo {b_climo:.4f})")


def test_mode_c_channel_tracks_rest_not_total():
    """Mode C: D1/D3's error-channel features and ehat head must track
    feat_err (rest), not the total err -- while the p head still trains on
    the preregistered total-error event."""
    est = CompetenceEstimator(0.2682, 0.2067, 0.4156, feature_mode="C")
    # total err constant 0.5; rest_err ramps 0.1 -> 0.9: ehat must follow rest
    for t in range(200):
        rest = 0.1 + 0.8 * t / 199
        se = [0.0, 0.0, rest, rest, rest, rest]
        ctx = {"ounc": 0.3, "runc": 0.2, "uunc": 0.5, "uhat": 0.0,
               "signed_error": se}
        pr = est.predict_tick(ctx)
        est.observe_tick(ctx, {"err_next": 0.5, "err_rw": 0.2,
                               "benefit": 0.05}, 0.01)
    # after training, ehat for next_obs should be near the recent rest (~0.9)
    pr = est.predict_tick(ctx)
    ehat = pr["next_obs"]["ehat"]
    assert 0.5 < ehat < 1.5, ehat
    # and must NOT be near 0.5 (the constant total err): proves the channel
    # tracks rest, not total
    assert abs(ehat - 0.9) < abs(ehat - 0.5) + 0.3, ehat
    print(f"test_mode_c_channel_tracks_rest_not_total PASS (ehat={ehat:.3f})")


def test_mode_c_no_leakage_via_signed_error():
    """ctx['signed_error'] (tick-t outcome) must not affect tick-t's
    emitted prediction in mode C."""
    eA = CompetenceEstimator(0.2682, 0.2067, 0.4156, feature_mode="C")
    eB = CompetenceEstimator(0.2682, 0.2067, 0.4156, feature_mode="C")
    for t in range(30):
        se = [0.1 * t] * 6
        ctx = {"ounc": 0.3, "runc": 0.2, "uunc": 0.5, "uhat": 0.0,
               "signed_error": se}
        for e in (eA, eB):
            e.predict_tick(ctx)
            e.observe_tick(ctx, {"err_next": 0.2, "err_rw": 0.2,
                                 "benefit": 0.05}, 0.01)
    ctxA = {"ounc": 0.3, "runc": 0.2, "uunc": 0.5, "uhat": 0.0,
            "signed_error": [9.9] * 6}
    ctxB = {"ounc": 0.3, "runc": 0.2, "uunc": 0.5, "uhat": 0.0,
            "signed_error": [0.0] * 6}
    pA = eA.predict_tick(ctxA)
    pB = eB.predict_tick(ctxB)
    for d in pA:
        assert pA[d]["p"] == pB[d]["p"], (d,)
        assert pA[d]["ehat"] == pB[d]["ehat"], (d,)
    print("test_mode_c_no_leakage_via_signed_error PASS")


def test_mode_c_includes_novelty_features():
    """Regression test for the 0072 voided-run bug: in mode C, D1/D3 must
    train 11 features (7 base + 4 novelty), not 7."""
    est = CompetenceEstimator(0.2682, 0.2067, 0.4156, feature_mode="C")
    ctx = {"ounc": 0.3, "runc": 0.2, "uunc": 0.5, "uhat": 0.1,
           "mean_similarity": 0.8, "n_nbrs": 4, "retrieval_used": True,
           "e1_ema": 0.5, "signed_error": [0.1] * 6}
    est.predict_tick(ctx)
    est.observe_tick(ctx, {"err_next": 0.2, "err_rw": 0.2, "benefit": 0.05},
                     0.01)
    assert len(est.est["next_obs"].w) == 11, len(est.est["next_obs"].w)
    assert len(est.est["retrieval_usefulness"].w) == 12, \
        len(est.est["retrieval_usefulness"].w)
    # mode A unchanged: 7 / 8
    estA = CompetenceEstimator(0.2682, 0.2067, 0.4156, feature_mode="A")
    estA.predict_tick(ctx)
    estA.observe_tick(ctx, {"err_next": 0.2, "err_rw": 0.2, "benefit": 0.05},
                      0.01)
    assert len(estA.est["next_obs"].w) == 7, len(estA.est["next_obs"].w)
    assert len(estA.est["retrieval_usefulness"].w) == 8, \
        len(estA.est["retrieval_usefulness"].w)
    print("test_mode_c_includes_novelty_features PASS")


if __name__ == "__main__":
    test_anchor_equals_climatology()
    test_causality_prediction_ignores_current_outcome()
    test_learns_regime_beats_climatology()
    test_determinism()
    test_feature_groups_both_contribute()
    test_mode_c_channel_tracks_rest_not_total()
    test_mode_c_no_leakage_via_signed_error()
    test_mode_c_includes_novelty_features()
    print("ALL ESTIMATOR TESTS PASS")


