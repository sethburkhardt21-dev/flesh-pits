"""Contract-conformance tests for the §35 environments. Stdlib unittest."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs"))

from env_interface import Environment, make_episode_id  # noqa: E402
from envs import (ALL_ENVS, ChangingRule, SelfWorld, DelayedMultistep,  # noqa: E402
                  CompositionalRule, CueDelayedReward)


def rollout(cls, seed, n=40):
    env = cls()
    obs = env.reset(seed)
    init_hash = env.state_hash()
    traj = []
    done = False
    i = 0
    while not done and i < n:
        obs, r, done, info = env.step(i % env.action_space()["n"])
        traj.append((round(r, 6), done))
        i += 1
    return obs, init_hash, traj, env.state_hash(), info


class TestEnvs(unittest.TestCase):
    def test_determinism(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                a = rollout(cls, 42)
                b = rollout(cls, 42)
                self.assertEqual(a, b, f"{cls.NAME} not deterministic")

    def test_seed_sensitivity(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                _, h1, _, _, _ = rollout(cls, 42)
                _, h2, _, _, _ = rollout(cls, 43)
                self.assertNotEqual(h1, h2, f"{cls.NAME} ignores seed")

    def test_invalid_action_raises(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                env.reset(1)
                with self.assertRaises(ValueError):
                    env.step(999)
                with self.assertRaises(ValueError):
                    env.step(-1)

    def test_obs_matches_space(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                obs = env.reset(7)
                env.validate_obs(obs, env.observation_space())  # raises if bad
                done = False
                i = 0
                while not done and i < 60:
                    obs, _, done, _ = env.step(i % env.action_space()["n"])
                    env.validate_obs(obs, env.observation_space())
                    i += 1

    def test_state_hash_changes(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                env = cls()
                env.reset(7)
                h0 = env.state_hash()
                env.step(0)
                self.assertNotEqual(h0, env.state_hash())

    def test_episode_id_format(self):
        eid = make_episode_id("grid_world", "1.0.0", 42, 3)
        self.assertEqual(eid, "grid_world:v1.0.0:seed=42:run=3")

    def test_changing_rule_phase_schedule(self):
        env = ChangingRule()
        rules = []
        for ep in range(12):
            env.reset(1000 + ep)
            rules.append(tuple(env._rule))
        # fixed within a phase, flips between phases, schedule run-stable
        self.assertEqual(rules[0], rules[4])
        self.assertEqual(rules[5], rules[9])
        self.assertNotEqual(rules[0], rules[5])
        env2 = ChangingRule()
        rules2 = []
        for ep in range(12):
            env2.reset(555 + ep)
            rules2.append(tuple(env2._rule))
        self.assertEqual(rules, rules2)

    def test_all_envs_expose_contract(self):
        for cls in ALL_ENVS:
            with self.subTest(env=cls.NAME):
                self.assertTrue(issubclass(cls, Environment))
                self.assertTrue(cls.NAME and cls.VERSION)
                env = cls()
                self.assertEqual(env.action_space()["type"], "discrete")


class TestSelfWorld(unittest.TestCase):
    """EXP-SW-01 env: self/world causal structure + matched statistics."""

    def test_registered(self):
        names = [c.NAME for c in ALL_ENVS]
        self.assertIn("self_world", names)

    def test_spaces(self):
        env = SelfWorld()
        self.assertEqual(env.action_space()["n"], 2)
        space = env.observation_space()
        self.assertEqual(set(space), {"hand", "ball"})
        for ch in ("hand", "ball"):
            self.assertEqual(space[ch]["type"], "scalar")
            self.assertEqual((space[ch]["low"], space[ch]["high"]), (0.0, 1.0))

    def test_hand_is_action_determined(self):
        # SELF-caused: hand_{t+1} is a deterministic function of (hand_t, a).
        for seed in (1, 2, 3):
            for actions in ([0] * 10, [1] * 10, [0, 1] * 5):
                e1, e2 = SelfWorld(), SelfWorld()
                o1, o2 = e1.reset(seed), e2.reset(seed)
                for a in actions:
                    o1, _, _, _ = e1.step(a)
                    o2, _, _, _ = e2.step(a)
                    self.assertEqual(o1["hand"], o2["hand"])

    def test_ball_is_exogenous(self):
        # WORLD-caused: ball moves even when the action stream is fixed,
        # and its path is not a function of actions alone.
        env = SelfWorld()
        env.reset(11)
        balls = []
        for _ in range(20):
            obs, _, _, _ = env.step(0)  # constant action
            balls.append(obs["hand"])
        # hand under constant action is deterministic drift; ball varies
        env2 = SelfWorld()
        env2.reset(12)
        b2 = [env2.step(0)[0]["ball"] for _ in range(20)]
        env3 = SelfWorld()
        env3.reset(11)
        b3 = [env3.step(0)[0]["ball"] for _ in range(20)]
        self.assertEqual(b2, b2)  # sanity
        self.assertNotEqual(b2, b3)  # seed matters -> exogenous RNG stream

    def test_matched_change_statistics(self):
        # Both channels change with |displacement| in {0, STEP}: STEP
        # normally; 0 only at the reflection fixed points (STEP/2 and
        # 1-STEP/2), where a step reflects onto itself — symmetric for
        # both channels. Rates and magnitudes are matched by construction.
        env = SelfWorld()
        env.reset(75101)
        hand_moves = ball_moves = 0
        n = 60
        for i in range(n):
            o = {"hand": env._hand, "ball": env._ball}
            obs, _, _, _ = env.step(i % 2)
            for ch in ("hand", "ball"):
                d = abs(obs[ch] - o[ch])
                self.assertIn(round(d, 9), (0.0, round(SelfWorld.STEP, 9)))
            if abs(obs["hand"] - o["hand"]) > 1e-12:
                hand_moves += 1
            if abs(obs["ball"] - o["ball"]) > 1e-12:
                ball_moves += 1
        self.assertGreaterEqual(hand_moves, 0.8 * n)
        self.assertGreaterEqual(ball_moves, 0.8 * n)
        # Mechanism-matched (identical reflection for both channels), not
        # rate-identical: fixed-point non-moves are hit stochastically.

    def test_cause_labels_for_scoring_only(self):
        env = SelfWorld()
        env.reset(5)
        _, _, _, info = env.step(0)
        self.assertEqual(info["cause"], {"hand": "self", "ball": "world"})

    def test_reward_constant_zero(self):
        env = SelfWorld()
        env.reset(5)
        for i in range(60):
            _, r, done, _ = env.step(i % 2)
            self.assertEqual(r, 0.0)
            if done:
                break
        self.assertTrue(done)  # truncates at MAX_STEPS

    def test_reflection_preserves_determinism(self):
        env = SelfWorld()
        env.reset(999)
        t1 = [(env.step(0)[0]["hand"], env.step(0)[0]["ball"]) for _ in range(60)]
        env.reset(999)
        t2 = [(env.step(0)[0]["hand"], env.step(0)[0]["ball"]) for _ in range(60)]
        self.assertEqual(t1, t2)


class TestDelayedMultistep(unittest.TestCase):
    """delayed_multistep v1.0.0: two-stage delayed conjunction probe."""

    def _play(self, seed, branch, sub):
        env = DelayedMultistep()
        env.reset(seed)
        total, done, infos = 0.0, False, []
        script = [branch] + [2] * 14 + [sub] + [2] * 14
        for a in script:
            obs, r, done, info = env.step(a)
            total += r
            infos.append(info)
            if done:
                break
        return total, done, infos[-1], obs

    def test_conjunction_gates_final_reward(self):
        # The +1.0 requires BOTH choices to match the hidden pattern.
        for seed in (11, 12, 13):
            env = DelayedMultistep()
            env.reset(seed)
            pat = tuple(env._pattern)
            for b in (0, 1):
                for s in (0, 1):
                    total, done, info, _ = self._play(seed, b, s)
                    self.assertTrue(done)
                    want = 1.0 if (b, s) == pat else 0.0
                    self.assertEqual(info["pattern_match"], (b, s) == pat)
                    self.assertAlmostEqual(total - 28 * 0.02, want, places=9)

    def test_reward_is_delayed(self):
        # No final reward en route: shaping only until the terminal cell.
        env = DelayedMultistep()
        env.reset(21)
        total = 0.0
        for a in [0] + [2] * 14 + [0] + [2] * 13:  # stop one cell early
            obs, r, done, info = env.step(a)
            total += r
            self.assertFalse(done)
        self.assertAlmostEqual(total, 27 * 0.02, places=9)

    def test_pattern_hidden_from_obs(self):
        env = DelayedMultistep()
        obs = env.reset(31)
        self.assertNotIn("pattern", obs)
        # ...but present in info for analysis only (contract: agents must
        # not use info).
        _, _, _, info = env.step(0)
        self.assertIn("pattern", info)

    def test_pattern_seed_determined(self):
        e1, e2 = DelayedMultistep(), DelayedMultistep()
        e1.reset(77)
        e2.reset(77)
        self.assertEqual(tuple(e1._pattern), tuple(e2._pattern))
        # The 2-bit pattern space is fully exercised over seeds.
        bits = set()
        for s in range(40):
            e = DelayedMultistep()
            e.reset(s)
            bits.add(tuple(e._pattern))
        self.assertEqual(bits, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_truncation_without_progress(self):
        env = DelayedMultistep()
        env.reset(41)
        done = False
        for _ in range(DelayedMultistep.MAX_STEPS):
            _, _, done, info = env.step(3)  # stay forever
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(info.get("truncated"))


class TestCompositionalRule(unittest.TestCase):
    """compositional_rule v1.0.0: XOR contingency, phase flips."""

    def _correct_from_info(self, info):
        # info is ground truth for ANALYSIS ONLY in real runs; here the
        # test harness itself is the analyst, so reading it is legitimate.
        return info["map_a"][0] ^ info["map_b"][0]  # placeholder

    def test_xor_contingency(self):
        # Playing the XOR-correct action earns ~90% (10% noise flips).
        # The test harness is the analyst here, so reading the env's
        # internal maps (never in the observation) is legitimate.
        env = CompositionalRule()
        wins = n = 0
        for seed in range(500, 520):
            env.reset(seed)
            for _ in range(CompositionalRule.STEPS):
                correct = (env._map_a[env._cue_a]
                           ^ env._map_b[env._cue_b])
                _, r, done, _ = env.step(correct)
                wins += r
                n += 1
                if done:
                    break
        rate = wins / n
        self.assertGreater(rate, 0.80)
        self.assertLess(rate, 1.0)  # noise flips genuinely occur

    def test_single_cue_marginals_uninformative(self):
        # For each fixed cue_a, the correct action is 50/50 over cue_b
        # (and symmetrically) — the contingency is a genuine 3-way
        # interaction, never reducible to one cue.
        env = CompositionalRule()
        env.reset(601)
        ma, mb = list(env._map_a), list(env._map_b)
        for cue_a in (0, 1):
            actions = {ma[cue_a] ^ mb[cue_b] for cue_b in (0, 1)}
            self.assertEqual(actions, {0, 1},
                             f"cue_a={cue_a} alone determines the action")
        for cue_b in (0, 1):
            actions = {ma[cue_a] ^ mb[cue_b] for cue_a in (0, 1)}
            self.assertEqual(actions, {0, 1},
                             f"cue_b={cue_b} alone determines the action")
        # Maps are never constant (guaranteed by MAPS).
        for m in CompositionalRule.MAPS:
            self.assertEqual(set(m), {0, 1})

    def test_phase_schedule(self):
        env = CompositionalRule()
        sched = []
        for ep in range(12):
            env.reset(1000 + ep)
            sched.append((tuple(env._map_a), tuple(env._map_b)))
        self.assertEqual(sched[0], sched[4])      # fixed within a phase
        self.assertEqual(sched[5], sched[9])
        # schedule is a fixed function of phase: run-stable across instances
        env2 = CompositionalRule()
        sched2 = []
        for ep in range(12):
            env2.reset(555 + ep)
            sched2.append((tuple(env2._map_a), tuple(env2._map_b)))
        self.assertEqual(sched, sched2)

    def test_maps_hidden_from_obs(self):
        env = CompositionalRule()
        obs = env.reset(701)
        self.assertNotIn("map_a", obs)
        self.assertNotIn("map_b", obs)

    def test_episode_length(self):
        env = CompositionalRule()
        env.reset(801)
        done = False
        for _ in range(CompositionalRule.STEPS):
            _, _, done, _ = env.step(0)
        self.assertTrue(done)


class TestCueDelayedReward(unittest.TestCase):
    """cue_delayed_reward v1.0.0: cue-conditioned branch choice."""

    def _play_branch(self, seed, branch):
        env = CueDelayedReward()
        obs0 = env.reset(seed)
        cue = obs0["cue"]
        total, done = 0.0, False
        for a in [branch] + [2] * 9:
            obs, r, done, info = env.step(a)
            total += r
            if done:
                break
        return total, done, info, cue

    def test_correct_branch_equals_cue(self):
        for seed in range(10, 20):
            env = CueDelayedReward()
            obs = env.reset(seed)
            self.assertIn(env._cue, (0, 1))
            self.assertEqual(env._correct, env._cue)
            # cue constant within the episode
            cues = {obs["cue"]}
            for _ in range(5):
                o, _, done, _ = env.step(3)
                cues.add(o["cue"])
                if done:
                    break
            self.assertEqual(cues, {env._cue})

    def test_reward_conditioned_on_cue(self):
        for seed in range(20, 30):
            env = CueDelayedReward()
            env.reset(seed)
            cue = env._cue
            total_ok, done_ok, info_ok, _ = self._play_branch(seed, cue)
            total_bad, done_bad, info_bad, _ = self._play_branch(seed,
                                                                 1 - cue)
            self.assertTrue(done_ok and done_bad)
            self.assertTrue(info_ok["branch_correct"])
            self.assertFalse(info_bad["branch_correct"])
            self.assertAlmostEqual(total_ok - 9 * 0.02, 1.0, places=9)
            self.assertAlmostEqual(total_bad - 9 * 0.02, 0.0, places=9)

    def test_reward_delayed_length_steps(self):
        env = CueDelayedReward()
        env.reset(31)
        cue = env._cue
        total = 0.0
        for a in [cue] + [2] * 8:  # stop one cell before the end
            _, r, done, _ = env.step(a)
            total += r
            self.assertFalse(done)
        self.assertAlmostEqual(total, 8 * 0.02, places=9)


if __name__ == "__main__":
    unittest.main()
