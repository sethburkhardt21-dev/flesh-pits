"""Unit tests — Architecture B prototype.

Stdlib unittest. Covers: determinism, contract compliance, precision
ablation hook, belief persistence, L1 lesion wiring, snapshot/restore,
prediction logging, memory retrieval, and the neutrality scan (no
identity-privileged strings anywhere in the prototype tree).
"""

import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
sys.path.insert(0, PROTO)

from precision import PrecisionEstimator  # noqa: E402
from generative_model import HierarchicalGenerativeModel  # noqa: E402
from memory import EpisodicStore  # noqa: E402
from predictions import PredictionLog, UsefulnessPredictor  # noqa: E402
from active_inference import ActiveInferenceSelector  # noqa: E402
from affect import ErrorAffect, PadController  # noqa: E402
from agent import ArchB  # noqa: E402


SPACE = {
    "cue": {"type": "categorical", "values": [0, 1]},
    "last_reward": {"type": "scalar", "low": 0.0, "high": 1.0},
}


def obs(cue, r=0.0):
    return {"cue": cue, "last_reward": r}


class TestPrecision(unittest.TestCase):
    def test_uniform_ablation_hook(self):
        import random
        rng = random.Random(0)
        p = PrecisionEstimator(3, uniform=True)
        pi = p.observe([0.5, -0.2, 0.1])
        self.assertEqual(pi, [1.0, 1.0, 1.0])
        p.set_uniform(False)
        for _ in range(60):
            p.observe([rng.gauss(0, 0.1), rng.gauss(0, 2.0),
                       rng.gauss(0, 0.1)])
        pi = p.current()
        # Noisy channel (index 1) must be down-weighted vs quiet channels.
        self.assertLess(pi[1], pi[0])
        self.assertLess(pi[1], pi[2])

    def test_estimated_not_handset(self):
        import random
        rng = random.Random(1)
        p = PrecisionEstimator(2, window=10)
        for _ in range(20):
            p.observe([rng.gauss(0, 0.05), rng.gauss(0, 1.0)])
        pi = p.current()
        self.assertGreater(pi[0], pi[1])
        self.assertNotEqual(pi[0], 1.0)


class TestGenerativeModel(unittest.TestCase):
    def _model(self, **kw):
        kw.setdefault("cat_slices", [(0, 2)])
        return HierarchicalGenerativeModel(6, 2, **kw)

    def test_beliefs_persist_across_ticks(self):
        m = self._model(seed=0)
        m.observe([1, 0, 0.0, 1, 0, 0], 0, (0,),
                  [1, 0, 0.9, 0, 1, 0], 1.0)
        # L1 experts created and non-trivial; mu0 moved off zero.
        self.assertGreater(m.n_contexts, 0)
        self.assertNotEqual(m.mu0, [0.0] * 6)
        D, R = m._experts((0,))
        self.assertTrue(any(abs(x) > 1e-9 for row in D for x in row)
                        or any(abs(x) > 1e-9 for x in R))

    def test_update_proportional_to_precision_weighted_error(self):
        m = self._model(seed=0)
        d = m.observe([1, 0, 0.0, 1, 0, 0], 0, (0,),
                      [1, 0, 1.0, 0, 1, 0], 0.0)
        self.assertEqual(len(d["e0"]), 6)
        self.assertEqual(len(d["pi0"]), 6)
        self.assertGreater(d["update_norm"], 0.0)

    def test_l1_lesion_disables_topdown(self):
        m = self._model(seed=3)
        for i in range(20):
            m.observe([1, 0, 0.1 * i, 1, 0, 0], i % 2, (0,),
                      [0, 1, 0.1 * i + 0.5, 0, 1, 0], 1.0)
        td = m.topdown(m.feat([1, 0, 0.5, 1, 0, 0], 0), (0,))
        self.assertGreater(sum(abs(x) for x in td), 1e-9)
        d = m.observe([1, 0, 0.5, 1, 0, 0], 0, (0,),
                      [0, 1, 1.0, 0, 1, 0], 1.0, lesion_l1=True)
        self.assertEqual(d["e1"], d["e0"])  # no top-down explanation
        # Lesioned reward prediction has no context term either.
        r_full = m.predict_reward([1, 0, 0.5, 1, 0, 0], 0, (0,))
        self.assertIsInstance(r_full, float)

    def test_context_experts_learn_contingency(self):
        # cue0->a0, cue1->a1 deterministic; random policy so both actions
        # are tried per cue: R table must separate good vs bad.
        import random
        rng = random.Random(0)
        m = self._model(seed=0)
        for i in range(400):
            cue = rng.randrange(2)
            a = rng.randrange(2)
            r = 1.0 if a == cue else 0.0
            o = ([1.0, 0.0] if cue == 0 else [0.0, 1.0]) + [0.0, 1, 0, 0]
            o2 = ([1.0, 0.0] if rng.randrange(2) == 0 else [0.0, 1.0])
            o2 += [r, 1.0, 0.0, 0.0]
            m.observe(o, a, (cue,), o2, r)
        for cue, good_a in ((0, 0), (1, 1)):
            o = ([1.0, 0.0] if cue == 0 else [0.0, 1.0]) + [0.0, 1, 0, 0]
            rg = m.predict_reward(o, good_a, (cue,))
            rb = m.predict_reward(o, 1 - good_a, (cue,))
            self.assertGreater(rg, rb,
                               f"cue {cue}: good {rg:.3f} vs bad {rb:.3f}")

    def test_snapshot_restore(self):
        m = self._model(seed=5)
        for i in range(10):
            m.observe([1, 0, 0.1, 1, 0, 0], 0, (0,),
                      [0, 1, 0.5, 0, 1, 0], 1.0)
        s = m.snapshot()
        json.dumps(s)  # must be JSON-serializable
        m2 = self._model(seed=5)
        m2.restore(s)
        self.assertEqual(m.predict_next([1, 0, 0.2, 1, 0, 0], 1, (0,)),
                         m2.predict_next([1, 0, 0.2, 1, 0, 0], 1, (0,)))
        self.assertEqual(m.n_contexts, m2.n_contexts)


class TestMemory(unittest.TestCase):
    def test_store_retrieve_provenance(self):
        mem = EpisodicStore()
        mid = mem.store([1.0, 0.0], 0, 1.0, [0.1, 0.0], [0.05, 0.0],
                        [1.0, 1.0], 0.9, tick=3, episode=1, env_name="t")
        self.assertTrue(mid.startswith("mem_"))
        self.assertEqual(mem.indexed_count, 1)
        nbrs = mem.retrieve([1.0, 0.0], k=1)
        self.assertEqual(len(nbrs), 1)
        self.assertEqual(nbrs[0]["exp"]["tick"], 3)
        self.assertEqual(nbrs[0]["exp"]["env"], "t")

    def test_dim_mismatch_scores_zero(self):
        mem = EpisodicStore()
        mem.store([1.0, 0.0], 0, 1.0, [0.1, 0.0], [0.05, 0.0],
                  [1.0, 1.0], 0.9, 0, 0, "t")
        nbrs = mem.retrieve([1.0, 0.0, 0.0], k=1)
        self.assertEqual(nbrs[0]["score"], 0.0)

    def test_below_threshold_not_indexed(self):
        mem = EpisodicStore()
        mid = mem.store([1.0], 0, 0.0, [0.0], [0.0], [1.0], 0.1, 0, 0, "t")
        self.assertEqual(mid, "")
        self.assertEqual(mem.indexed_count, 0)
        self.assertEqual(len(mem), 1)  # still in the recent list


class TestAffect(unittest.TestCase):
    def test_error_affect_valence_sign(self):
        a = ErrorAffect()
        for _ in range(40):  # warm up: EMAs converge near e=1.0
            a.update(1.0, 0.1)
        # Declining error -> positive valence.
        for e in (0.8, 0.6, 0.4, 0.2, 0.1, 0.05):
            k = a.update(e, 0.1)
        self.assertGreater(k["valence"], 0.0)
        # Rising error -> negative valence.
        b = ErrorAffect()
        for _ in range(40):
            b.update(0.05, 0.1)
        for e in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
            k = b.update(e, 0.1)
        self.assertLess(k["valence"], 0.0)

    def test_pad_baseline_runs(self):
        p = PadController()
        k = p.update(reward=1.0, novelty=0.5, stress=0.2)
        self.assertIn("explore_gain", k)
        self.assertGreater(k["valence"], 0.0)


class TestAgent(unittest.TestCase):
    def _agent(self, **kw):
        return ArchB(observation_space=SPACE, n_actions=2,
                     env_name="test", seed=0, **kw)

    def test_act_update_contract(self):
        a = self._agent()
        a.reset(123, {"type": "discrete", "n": 2})
        act0 = a.act(obs(0))
        self.assertIn(act0, (0, 1))
        a.update(obs(0), act0, 1.0, False, {})
        act1 = a.act(obs(1, 1.0))
        self.assertIn(act1, (0, 1))

    def test_determinism(self):
        def run():
            a = self._agent()
            a.reset(7, {"type": "discrete", "n": 2})
            acts = []
            o = obs(0)
            for i in range(20):
                act = a.act(o)
                acts.append(act)
                a.update(o, act, float(i % 2), False, {})
                o = obs(i % 2, float(i % 2))
            return acts, a.snapshot()
        acts1, snap1 = run()
        acts2, snap2 = run()
        self.assertEqual(acts1, acts2)
        self.assertEqual(snap1["model"]["ctx_R"], snap2["model"]["ctx_R"])

    def test_learning_reduces_error_on_structure(self):
        a = self._agent()
        a.reset(9, {"type": "discrete", "n": 2})
        # Structured: cue predicts which action gives reward 1.
        errs = []
        o = obs(0)
        for i in range(60):
            cue = i % 2
            o = obs(cue)
            act = a.act(o)
            r = 1.0 if act == cue else 0.0
            a.update(o, act, r, False, {})
            errs.append(a.agg["e0_abs"][-1] if a.agg["e0_abs"] else 0.0)
        early = sum(errs[:10]) / 10
        late = sum(errs[-10:]) / 10
        self.assertLess(late, early)

    def test_frozen_learns_nothing(self):
        a = self._agent(frozen=True)
        w0 = [row[:] for row in a.model.W]
        a.reset(9, {"type": "discrete", "n": 2})
        o = obs(0)
        for i in range(10):
            act = a.act(o)
            a.update(o, act, 1.0, False, {})
            o = obs(1, 1.0)
        self.assertEqual(a.model.W, w0)

    def test_snapshot_restore_roundtrip(self):
        a = self._agent()
        a.reset(11, {"type": "discrete", "n": 2})
        o = obs(0)
        for i in range(8):
            act = a.act(o)
            a.update(o, act, 1.0, False, {})
            o = obs(1, 1.0)
        s = a.snapshot()
        json.dumps(s)
        b = self._agent()
        b.restore(s)
        v = [1.0, 0.0, 1.0]
        self.assertEqual(
            a.model.predict_next(v, 0, (0,)),
            b.model.predict_next(v, 0, (0,)))

    def test_prediction_log_writes(self):
        import tempfile
        with tempfile.NamedTemporaryFile(
                suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            a = self._agent(log_path=path)
            a.reset(13, {"type": "discrete", "n": 2})
            o = obs(0)
            for i in range(5):
                act = a.act(o)
                a.update(o, act, 1.0, False, {})
                o = obs(1, 1.0)
            a.plog.close()
            with open(path) as fh:
                lines = fh.readlines()
            # 4 closed ticks x 3 targets = 12 records.
            self.assertEqual(len(lines), 12)
            rec = json.loads(lines[0])
            for key in ("prediction", "confidence", "uncertainty", "actual",
                        "signed_error", "abs_error", "update_norm"):
                self.assertIn(key, rec)
        finally:
            os.unlink(path)

    def test_neutrality_scan(self):
        bad = ["seth", "master", "darklord", "dark_lord", "worship",
               "devot", "obey", "erin"]
        hits = []
        for fn in os.listdir(PROTO):
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(PROTO, fn)) as fh:
                text = fh.read().lower()
            for token in bad:
                if token in text:
                    hits.append((fn, token))
        self.assertEqual(hits, [])


class TestSelector(unittest.TestCase):
    def test_modes(self):
        s = ActiveInferenceSelector(3, mode="greedy", seed=0)
        a = s.select(lambda a: float(a), lambda a: 0.0, 0.0)
        self.assertEqual(a, 2)  # argmax reward
        s2 = ActiveInferenceSelector(3, mode="random", seed=0)
        seen = {s2.select(lambda a: 0.0, lambda a: 0.0, 0.0)
                for _ in range(20)}
        self.assertGreater(len(seen), 1)


class TestUsefulness(unittest.TestCase):
    def test_learns_benefit(self):
        u = UsefulnessPredictor()
        q = u.features(0.9, 5, 0.2, 0.1)
        for _ in range(50):
            u.update(q, 0.5)
        self.assertAlmostEqual(u.predict(q), 0.5, delta=0.1)


class TestShadowGate(unittest.TestCase):
    """EXP-FP-0007 instrument: the gate reads a shadow-trained predictor."""

    def _agent(self, **kw):
        kw.setdefault("gate_uhat_source", "shadow")
        return ArchB(observation_space=SPACE, n_actions=2,
                     env_name="test", seed=0, **kw)

    def _stub_correction(self, agent, corr_vec=(0.5, 0.0, 0.0)):
        agent.memory.retrieval_correction = lambda ov, k=5: {
            "correction": list(corr_vec),
            "mean_similarity": 0.9, "n": 5}

    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            ArchB(observation_space=SPACE, n_actions=2, env_name="test",
                  gate_uhat_source="bogus")

    def test_live_is_default_and_shadow_zero_init(self):
        a = ArchB(observation_space=SPACE, n_actions=2, env_name="test")
        self.assertEqual(a.gate_uhat_source, "live")
        self.assertEqual(a.shadow_usefulness.w, [0.0] * 4)
        self.assertEqual(a.shadow_usefulness.b, 0.0)

    def test_gate_reads_shadow_not_live(self):
        # The EXP-FP-0006 degeneracy: live uhat is cold at exactly 0.0 and
        # the strict sign gate would block. In shadow mode the gate must
        # read the shadow predictor instead.
        a = self._agent(gate_policy="uhat", gate_threshold=0.0)
        a.reset(5, {"type": "discrete", "n": 2})
        a.shadow_usefulness.w = [2.0, 0.0, 0.0, 0.0]
        self._stub_correction(a)
        a._open_tick([1.0, 0.0, 0.0], 0)
        st = a.gate.stats()
        self.assertEqual(st["n_available"], 1)
        # qfeat[0] = mean_similarity = 0.9 -> shadow uhat = 1.8 > 0.
        self.assertEqual(st["n_applied"], 1)
        self.assertGreater(a._pending["uhat_gate"], 0.0)
        self.assertEqual(a._pending["uhat"], 0.0)  # live still cold

    def test_shadow_trains_on_counterfactual(self):
        # Blocked correction: live predictor trains on realized benefit 0
        # (stays degenerate); shadow trains on the counterfactual
        # unconditional benefit (moves off zero).
        a = self._agent(gate_policy="uhat", gate_threshold=0.0)
        a.reset(5, {"type": "discrete", "n": 2})
        self._stub_correction(a)
        a._open_tick([1.0, 0.0, 0.0], 0)
        # Shadow is cold -> uhat_gate == 0.0 -> strict gate blocks.
        self.assertEqual(a.gate.stats()["n_applied"], 0)
        xhat_raw = a._pending["xhat_raw"]
        corr = a._pending["correction"]
        obs_helped = [x + c for x, c in zip(xhat_raw, corr)]
        a._close_tick(obs_helped)
        self.assertEqual(len(a._shadow_hist), 1)
        uhat_gate, benefit = a._shadow_hist[0]
        self.assertEqual(uhat_gate, 0.0)
        # Counterfactual benefit = |e0_raw| - 0 = |correction|.
        self.assertAlmostEqual(benefit, 0.5, places=9)
        self.assertNotEqual(a.shadow_usefulness.w, [0.0] * 4)
        self.assertNotEqual(a.shadow_usefulness.b, 0.0)
        # Live predictor saw realized benefit 0 -> unchanged (degenerate).
        self.assertEqual(a.usefulness.w, [0.0] * 4)
        self.assertEqual(a.usefulness.b, 0.0)

    def test_shadow_snapshot_restore(self):
        a = self._agent()
        a.reset(5, {"type": "discrete", "n": 2})
        a.shadow_usefulness.w = [1.0, 2.0, 3.0, 4.0]
        a.shadow_usefulness.b = 0.5
        s = a.snapshot()
        json.dumps(s)
        b = self._agent()
        b.restore(s)
        self.assertEqual(b.shadow_usefulness.w, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(b.shadow_usefulness.b, 0.5)
        self.assertEqual(b.gate_uhat_source, "shadow")

    def test_restore_tolerates_pre_shadow_snapshot(self):
        a = self._agent()
        a.reset(5, {"type": "discrete", "n": 2})
        s = a.snapshot()
        del s["shadow_usefulness"]
        del s["gate_uhat_source"]
        b = self._agent()
        b.restore(s)  # must not raise
        self.assertEqual(b.gate_uhat_source, "live")


if __name__ == "__main__":
    unittest.main(verbosity=2)
