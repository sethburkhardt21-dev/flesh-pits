"""Unit tests for the EXP-FP-0012 retention guard (additive).

Covers: full-rate registration on positive trials, kappa-scaled erasure
on non-positive trials, etaR save/restore (no leak), config parity of
the swapped agent, snapshot/restore round-trip of guard params, and
determinism.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generative_model import HierarchicalGenerativeModel
from retention_guard import RetentionGuardModel, make_retention_agent


def _twin_models(seed=7, kappa=20.0):
    kw = dict(obs_dim=5, n_actions=4, cat_slices=(), seed=seed)
    return (HierarchicalGenerativeModel(**kw),
            RetentionGuardModel(kappa=kappa, goal_thresh=0.0, **kw))


def test_positive_trial_full_rate():
    # From identical state, a positive trial must move the R table
    # byte-identically on the guarded and unguarded models (registration
    # untouched).
    base, guarded = _twin_models()
    ctx, s, a = (), [1.0, 0, 0, 1, 0.66], 2
    d_base = base.observe(s, a, ctx, s, 1.0)
    d_guard = guarded.observe(s, a, ctx, s, 1.0)
    assert abs(base.ctx_R[ctx][a] - guarded.ctx_R[ctx][a]) < 1e-12
    assert abs(d_base["rerr"] - d_guard["rerr"]) < 1e-12


def test_nonpositive_trial_scaled_rate():
    base, guarded = _twin_models()
    ctx, s, a = (), [1.0, 0, 0, 1, 0.66], 2
    # put credit in the table first (positive trial, full rate on both)
    base.observe(s, a, ctx, s, 1.0)
    guarded.observe(s, a, ctx, s, 1.0)
    r0_base = base.ctx_R[ctx][a]
    r0_guard = guarded.ctx_R[ctx][a]
    # one erasing -0.01 trial on each
    base.observe(s, a, ctx, s, -0.01)
    guarded.observe(s, a, ctx, s, -0.01)
    erase_base = r0_base - base.ctx_R[ctx][a]
    erase_guard = r0_guard - guarded.ctx_R[ctx][a]
    # piR differs slightly between the twins after the goal trial? No:
    # histories are identical, so piR is identical -> exact 1/kappa ratio.
    assert abs(erase_guard - erase_base / 20.0) < 1e-12, (
        erase_base, erase_guard)


def test_etaR_restored_no_leak():
    _, guarded = _twin_models(kappa=20.0)
    before = guarded.etaR
    guarded.observe([0.0] * 5, 1, (), [0.0] * 5, -0.01)
    assert guarded.etaR == before
    guarded.observe([0.0] * 5, 1, (), [0.0] * 5, 1.0)
    assert guarded.etaR == before


def test_zero_reward_counts_as_nonpositive():
    # A guarded model on an r=0.0 trial must behave exactly like an
    # unguarded twin whose etaR was manually divided by kappa (piR is
    # identical because the error histories match).
    kw = dict(obs_dim=5, n_actions=4, cat_slices=(), seed=21)
    guarded = RetentionGuardModel(kappa=20.0, goal_thresh=0.0, **kw)
    slow = HierarchicalGenerativeModel(**kw)
    slow.etaR = slow.etaR / 20.0
    s = [1.0, 0, 0, 1, 0.66]
    for t in range(5):
        guarded.observe(s, t % 4, (), s, -0.01)
        slow.observe(s, t % 4, (), s, -0.01)
    guarded.observe(s, 3, (), s, 0.0)
    slow.observe(s, 3, (), s, 0.0)
    assert abs(guarded.ctx_R[()][3] - slow.ctx_R[()][3]) < 1e-12
    assert guarded.etaR == 0.10  # restored to the constructor default


def test_agent_swap_config_parity():
    obs_space = {"wall": {"type": "vector", "shape": 4},
                 "beacon": {"type": "scalar"}}
    ag = make_retention_agent(obs_space, 4, env_name="pomaze", seed=11,
                              kappa=20.0)
    from agent import ArchB
    ref = ArchB(obs_space, 4, env_name="pomaze", seed=11)
    assert isinstance(ag.model, RetentionGuardModel)
    assert ag.model.obs_dim == ref.model.obs_dim == 5
    assert ag.model.n_actions == ref.model.n_actions == 4
    assert list(ag.model.cat_slices) == list(ref.model.cat_slices)
    assert ag.model.seed == ref.model.seed == 11
    assert ag.model.precision_kind == ref.model.precision_kind
    assert (ag.model.eta0, ag.model.etaD, ag.model.eta_r, ag.model.etaR) == \
        (ref.model.eta0, ref.model.etaD, ref.model.eta_r, ref.model.etaR)
    assert ag.base_etas == (ag.model.eta0, ag.model.etaD,
                            ag.model.eta_r, ag.model.etaR)
    assert ag.model.kappa == 20.0 and ag.model.goal_thresh == 0.0
    # frozen parity
    agf = make_retention_agent(obs_space, 4, env_name="pomaze", seed=11,
                               frozen=True)
    assert agf.model.etaR == 0.0 and agf.base_etas[3] == 0.0


def test_snapshot_restore_roundtrip():
    # Guard params persist through snapshot/restore. Uses a non-empty
    # context (categorical slice) because the BASE class restore() has a
    # pre-existing bug on the empty-context key: snapshot() serializes
    # ctx=() as "" and restore() does int("") -> ValueError. That bug is
    # in generative_model.py (core, untouched per additivity); this test
    # covers only the additive guard-param persistence.
    kw = dict(obs_dim=5, n_actions=4, cat_slices=[(0, 2)], seed=7)
    guarded = RetentionGuardModel(kappa=50.0, goal_thresh=0.25, **kw)
    s = [1.0, 0.0, 0.0, 1.0, 0.66]
    ctx = guarded.context_key(s)
    assert ctx != ()
    guarded.observe(s, 2, ctx, s, 1.0)
    snap = guarded.snapshot()
    fresh = RetentionGuardModel(kappa=20.0, cat_slices=[(0, 2)],
                                obs_dim=5, n_actions=4, seed=99)
    assert fresh.kappa == 20.0  # constructor default differs
    fresh.restore(snap)
    assert fresh.kappa == 50.0 and fresh.goal_thresh == 0.25
    assert abs(fresh.ctx_R[ctx][2] - guarded.ctx_R[ctx][2]) < 1e-12
    # pre-guard snapshot tolerance
    plain = HierarchicalGenerativeModel(**kw).snapshot()
    fresh.restore(plain)
    assert fresh.kappa == 20.0 and fresh.goal_thresh == 0.0


def test_determinism():
    def run_once():
        _, g = _twin_models(seed=3, kappa=20.0)
        s = [0.2, 0.4, 0.6, 0.8, 1.0]
        for t in range(30):
            g.observe(s, t % 4, (), s, -0.01 if t != 15 else 1.0)
        return g.predict_reward(s, 1, ())
    assert run_once() == run_once()
