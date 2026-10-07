"""Retention-guard wrapper — EXP-FP-0012-RETENTION (additive).

EXP-FP-0010 diagnosed arch-B's pomaze failure as credit-RETENTION: goal
trials register transiently (R_ctx[a*] += etaR*piR*rerr ~= +0.94) but the
subsequent -0.01 trials that re-use action a* produce
rerr ~= -0.01 - R[a*] — a self-erasing leaky integrator that drives the
per-context R-table entry back to ~0 within ~50 ticks. The global head
(w_r/b_r) tracks the -0.01 background mean by design and never held goal
credit; the episodic store is unbounded and never queried by reward
prediction, so it is not the washout locus (mechanism measured in
synthetic replay; see experiments/preregistration_RETENTION.json).

Intervention (ONE clean retention mechanism): an ASYMMETRIC R-table
learning rate. Positive-reward trials (goal trials) learn at the full
etaR; non-positive trials (the -0.01 background) learn at etaR/kappa.
This lengthens the erasure half-life ~kappa-fold while leaving
registration and background-mean tracking intact.

Additivity: prototypes/architecture-b/{agent,generative_model,memory,
precision}.py are UNTOUCHED (sha256 verified G0). This module subclasses
HierarchicalGenerativeModel (overriding observe() only) and provides a
factory that swaps the guarded model into a stock ArchB at construction
(the stock model is fresh — no learned state is lost).

Deterministic given the seed. Stdlib only. No identity strings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from generative_model import HierarchicalGenerativeModel


class RetentionGuardModel(HierarchicalGenerativeModel):
    """Hierarchical generative model with an asymmetric R-table rate.

    On trials with reward > goal_thresh the per-context R-table updates
    at the full etaR (registration untouched). On trials with reward <=
    goal_thresh (the -0.01 background stream that washes goal credit out)
    etaR is divided by kappa for the duration of observe() and then
    restored — the erasure leg is slowed ~kappa-fold. eta0/etaD/eta_r,
    the precision estimators, and the posterior belief update are
    untouched.
    """

    def __init__(self, *args, kappa: float = 20.0,
                 goal_thresh: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if kappa < 1.0:
            raise ValueError("kappa must be >= 1.0")
        self.kappa = float(kappa)
        self.goal_thresh = float(goal_thresh)

    def observe(self, state_vec: List[float], action: int,
                ctx, obs_next: List[float], reward: float,
                lesion_l1: bool = False) -> Dict:
        saved_etaR = self.etaR
        if reward <= self.goal_thresh:
            self.etaR = saved_etaR / self.kappa
        try:
            return super().observe(state_vec, action, ctx, obs_next,
                                   reward, lesion_l1=lesion_l1)
        finally:
            self.etaR = saved_etaR

    # -- persistence (additive; tolerant of pre-guard snapshots) -----------
    def snapshot(self) -> Dict:
        state = super().snapshot()
        state["retention_guard"] = {"kappa": self.kappa,
                                    "goal_thresh": self.goal_thresh}
        return state

    def restore(self, state: Dict) -> None:
        super().restore(state)
        rg = state.get("retention_guard", {})
        self.kappa = float(rg.get("kappa", 20.0))
        self.goal_thresh = float(rg.get("goal_thresh", 0.0))


def make_retention_agent(observation_space: dict, n_actions: int,
                         env_name: str, seed: int,
                         kappa: float = 20.0, goal_thresh: float = 0.0,
                         **archb_kwargs):
    """Build an ArchB whose generative model is a RetentionGuardModel.

    The stock ArchB is constructed first with the caller's kwargs, then
    its model is replaced by a RetentionGuardModel built with the
    byte-identical configuration (dims, cat slices, etas incl. frozen,
    precision kind, surprise window, seed). The stock model is fresh at
    construction, so the swap loses no learned state. base_etas is
    refreshed so ArchB._close_tick's save/restore cycle stays consistent.
    """
    from agent import ArchB  # local import: keeps module import-light
    agent = ArchB(observation_space, n_actions, env_name=env_name,
                  seed=seed, **archb_kwargs)
    old = agent.model
    etas = (dict(eta0=0.0, etaD=0.0, eta_r=0.0, etaR=0.0)
            if agent.frozen else {})
    guarded = RetentionGuardModel(
        old.obs_dim, agent.n_actions, cat_slices=list(old.cat_slices),
        uniform_precision=agent.uniform_precision,
        precision_kind=old.precision_kind,
        surprise_k=old.surprise_k, seed=old.seed,
        kappa=kappa, goal_thresh=goal_thresh, **etas)
    agent.model = guarded
    agent.base_etas = (guarded.eta0, guarded.etaD,
                       guarded.eta_r, guarded.etaR)
    return agent
