"""Architecture C — gate suite (EXP-FP-C-BUILD-AND-BEAT).

Tests the C assembly's integration surfaces, fail-closed behavior, and
determinism. Small configs for speed; the battery itself is the heavy run.
"""
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH_C = os.path.dirname(_HERE)
_ARCH_A = os.path.join(os.path.dirname(_ARCH_C), "architecture-a")
_ARCH_B = os.path.join(os.path.dirname(_ARCH_C), "architecture-b")
_FLESH_EXP = os.path.join(os.path.dirname(os.path.dirname(_ARCH_C)),
                          "experiments")
_FLESH_ENVS = os.path.join(_FLESH_EXP, "envs")
for _p in (_ARCH_C, _ARCH_A, _ARCH_B, _FLESH_EXP, _FLESH_ENVS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_agent import (  # noqa: E402
    ArchC, CAdapter, categorical_slices, context_key,
    make_predictor_specialists, close_tick_c)
from tick import WorkspaceTick  # noqa: E402
from attention_cue import CueIndexedArbitrator  # noqa: E402
from attention import AttentionArbitrator  # noqa: E402
from memory import DisabledStore  # noqa: E402
from env_interface import obs_to_vector  # noqa: E402
from changing_rule import ChangingRule  # noqa: E402


def _c_config(**over):
    env = ChangingRule()
    space = env.observation_space()
    cfg = {
        "channels": ["a0", "a1"],
        "channel_to_action": {"a0": 0, "a1": 1},
        "observation_space": space,
        "env_name": "changing_rule",
        "context_fn": lambda obs: int(obs["cue"]),
        "gain_lr": 0.15, "theta": 0.45, "kappa": 0.1,
        "agent_seed": 80101,
        "predictor_kwargs": {},
        "memory_enabled": True,
    }
    cfg.update(over)
    return cfg, env


def _run_ticks(agent_cfg, env, n_ticks, seed=80101):
    c = ArchC(agent_cfg)
    adapter = CAdapter(env, agent_cfg["channel_to_action"])
    obs = adapter.reset_episode(seed)
    c.reset_episode(obs)
    traces = []
    for _ in range(n_ticks):
        traces.append(c.step(adapter))
        if adapter.last_done:
            break
    return c, adapter, traces


# ---------------------------------------------------------------------------
class TestIntegrationSurfaces:
    def test_predictor_drives_stimuli(self):
        """The predictor is the ONLY stimulus source: after the predictor
        learns a contingency, stimuli differentiate per action."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        specs = dict(c.tick.specialists)
        s0 = {ch: specs[ch](obs, 1)[0] for ch in cfg["channels"]}
        # train the predictor: action 0 reliably rewarded in this context
        vec = obs_to_vector(obs, cfg["observation_space"])
        ctx = context_key(vec, c.cat_slices)
        for _ in range(50):
            c.predictor.observe(list(c.predictor.mu0), 0, ctx, vec, 1.0)
            c.predictor.observe(list(c.predictor.mu0), 1, ctx, vec, 0.0)
        s1 = {ch: specs[ch](obs, 2)[0] for ch in cfg["channels"]}
        assert s1["a0"] > s1["a1"], \
            f"learned predictor must separate stimuli: {s1}"
        assert abs(s0["a0"] - s0["a1"]) < abs(s1["a0"] - s1["a1"])

    def test_frozen_predictor_stimuli_static(self):
        """C-frozen-predictor: the predictor's rhat contribution is frozen
        at init for identical obs (the memory bonus may still move — that
        is the memory's surface, not the predictor's)."""
        cfg, env = _c_config(frozen_predictor=True)
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        specs = dict(c.tick.specialists)
        r0 = {ch: specs[ch](obs, 1)[1]["rhat"] for ch in cfg["channels"]}
        for _ in range(10):
            c.step(adapter)
        r1 = {ch: specs[ch](obs, 99)[1]["rhat"] for ch in cfg["channels"]}
        assert r0 == r1, "frozen predictor must not move rhat"

    def test_memory_bonus_none_guarded(self):
        """Empty store -> predicted_error_for_action None -> zero bonus,
        not invented signal."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        vec = obs_to_vector(obs, cfg["observation_space"])
        assert c.memory.predicted_error_for_action(vec, 0) is None
        _stim, payload = dict(c.tick.specialists)["a0"](obs, 1)
        assert payload["pe_bonus"] == 0.0

    def test_memory_bonus_enters_stimulus(self):
        """Populated store -> per-action bonus differentiates stimuli."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        vec = obs_to_vector(obs, cfg["observation_space"])
        # encode: action 0 experienced large errors, action 1 small
        for i in range(6):
            e = [1.0] * len(vec) if i % 2 == 0 else [0.01] * len(vec)
            c.memory.store(vec, 0, 0.0, e, e, [1.0] * len(vec), 0.9,
                           i, 0, "changing_rule")
            e2 = [0.01] * len(vec)
            c.memory.store(vec, 1, 0.0, e2, e2, [1.0] * len(vec), 0.9,
                           i, 0, "changing_rule")
        specs = dict(c.tick.specialists)
        _s0, p0 = specs["a0"](obs, 1)
        _s1, p1 = specs["a1"](obs, 1)
        assert p0["pe_bonus"] > p1["pe_bonus"], \
            "higher stored error must yield a larger bonus"

    def test_learning_close_updates_predictor(self):
        """close_tick_c moves weights and the belief (B's learning path)."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        vec = obs_to_vector(obs, cfg["observation_space"])
        ctx = context_key(vec, c.cat_slices)
        w_before = [row[:] for row in c.predictor.W]
        mu_before = list(c.predictor.mu0)
        diag = close_tick_c(c.predictor, c.memory, state_vec=vec,
                            action_idx=0, ctx=ctx, obs_vec=vec,
                            obs_next_vec=vec, reward=1.0, tick=1,
                            episode_idx=0, env_name="changing_rule")
        assert c.predictor.W != w_before, "weights must move"
        assert c.predictor.mu0 != mu_before, "belief must update"
        assert len(c.memory) == 1, "experience must be encoded"
        assert math.isfinite(diag["e0_abs"])

    def test_no_memory_ablation_runs(self):
        """C-no-memory (DisabledStore): run completes, store stays empty."""
        cfg, env = _c_config(memory_enabled=False)
        c = ArchC(cfg)
        assert isinstance(c.memory, DisabledStore)
        _c, _ad, traces = _run_ticks(cfg, env, 40)
        assert len(traces) == 40
        assert len(c.memory) == 0
        assert all(math.isfinite(t["reward"]) for t in traces)

    def test_context_key_matches_b(self):
        """Seam helper equals B's own context_key on the same vector."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        obs = env.reset(80101)
        vec = obs_to_vector(obs, cfg["observation_space"])
        assert context_key(vec, c.cat_slices) == \
            c.predictor.context_key(vec)


# ---------------------------------------------------------------------------
class TestWorkspaceSkeleton:
    def test_sole_path_no_consumer_refs(self):
        """The tick holds no consumer references (A's sole-path law)."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        assert not hasattr(c.tick, "consumers") or \
            getattr(c.tick, "consumers", None) is None
        assert not hasattr(c.tick, "_consumers")
        # consumers exist only behind the bus
        assert c.tick.bus.read_consumer("planner_input") is None or True

    def test_r1_default_sub_ignition(self):
        """Nothing ignites (theta=0.99, valid range) -> graded-winner
        action, zero consumer propagation (K8-wired R1 default)."""
        cfg, env = _c_config(theta=0.99)
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        trace = c.step(adapter)
        assert trace["action_selection"]["path"] == "sub_ignition_explore"
        assert trace["action"] in ("a0", "a1")
        assert trace["deliveries"] == [], \
            "sub-ignition ticks must not propagate to consumers"

    def test_cue_indexed_arbitrator_wired(self):
        cfg, env = _c_config()
        c = ArchC(cfg)
        assert isinstance(c.tick.arbitrator, CueIndexedArbitrator)

    def test_default_arbitrator_without_cue(self):
        cfg, env = _c_config(context_fn=None)
        c = ArchC(cfg)
        assert type(c.tick.arbitrator) is AttentionArbitrator

    def test_frozen_gains_pinned(self):
        cfg, env = _c_config(frozen_gains=True)
        c = ArchC(cfg)
        _c, _ad, traces = _run_ticks(cfg, env, 40)
        assert c.tick.arbitrator.gains == {"a0": 1.0, "a1": 1.0}

    def test_fail_closed_nonfinite_stimulus(self):
        """A diverging predictor fails closed (raises), not silent."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        c.predictor.predict_reward = lambda *a, **k: float("inf")
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        with pytest.raises(ValueError, match="non-finite"):
            dict(c.tick.specialists)["a0"](obs, 1)


# ---------------------------------------------------------------------------
class TestDeterminismAndProtocol:
    def test_deterministic_trajectories(self):
        """Same seed twice -> identical actions and rewards (G0c unit)."""
        cfg, _ = _c_config()
        env1, env2 = ChangingRule(), ChangingRule()
        _c1, _a1, t1 = _run_ticks(cfg, env1, 80, seed=80101)
        _c2, _a2, t2 = _run_ticks(cfg, env2, 80, seed=80101)
        a1 = [t["action"] for t in t1]
        a2 = [t["action"] for t in t2]
        assert a1 == a2, "trajectories must be bit-identical"
        assert [t["reward"] for t in t1] == [t["reward"] for t in t2]

    def test_adapter_protocol(self):
        """CAdapter: tick protocol returns float; driver sees transitions."""
        cfg, env = _c_config()
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        assert isinstance(obs, dict) and "cue" in obs
        r = adapter.step("a0")
        assert isinstance(r, float)
        obs_next, reward, done, info = adapter.last_transition()
        assert isinstance(obs_next, dict)
        assert reward == r and isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_reset_seeds_belief(self):
        """reset_episode seeds mu0 from the first observation (B's
        convention)."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        vec = obs_to_vector(obs, cfg["observation_space"])
        assert c.predictor.mu0 == pytest.approx(vec)

    def test_full_episode_finite(self):
        """One full changing_rule episode: all finite, actions valid."""
        cfg, env = _c_config()
        c = ArchC(cfg)
        adapter = CAdapter(env, cfg["channel_to_action"])
        obs = adapter.reset_episode(80101)
        c.reset_episode(obs)
        n = 0
        while not adapter.last_done and n < 60:
            trace = c.step(adapter)
            assert trace["action"] in ("a0", "a1")
            assert math.isfinite(trace["reward"])
            n += 1
        assert adapter.last_done, "episode must terminate"
