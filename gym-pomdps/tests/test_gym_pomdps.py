import itertools as itt
import unittest

import gym

import gym_pomdps


class Gym_POMDP_Test(unittest.TestCase):
    def test_list_pomdps(self):
        pomdps = gym_pomdps.list_pomdps()

        self.assertTrue(len(pomdps) > 0)
        for pomdp in pomdps:
            self.assertTrue(pomdp.startswith('POMDP'))

    def test_functional(self):
        env = gym.make('POMDP-shopping_2-continuing-v0')
        self._test_functional(env)
        env = gym.make('POMDP-shopping_2-episodic-v0')
        self._test_functional(env)
        env = gym.make('POMDP-arrowtrail-continuing-v0')
        self._test_run(env)

    def _test_functional(self, env):
        for _ in range(20):
            s = env.reset_functional()
            self.assertIsInstance(s, int)
            self.assertTrue(0 <= s < env.state_space.n)

            for s, a in itt.product(
                range(env.state_space.n), range(env.action_space.n)
            ):
                s1, o, r, done, info = env.step_functional(s, a)
                if done:
                    self.assertIsInstance(s1, int)
                    self.assertIsInstance(o, int)
                    self.assertIsInstance(r, float)
                    self.assertIsInstance(done, bool)
                    self.assertEqual((s1, done), (-1, True))
                    self.assertTrue(0 <= o < env.observation_space.n)
                    self.assertIn(r, env.rewards_dict.keys())
                    self.assertTrue(info is None or isinstance(info, dict))
                else:
                    self.assertIsInstance(s1, int)
                    self.assertIsInstance(o, int)
                    self.assertIsInstance(r, float)
                    self.assertIsInstance(done, bool)
                    self.assertTrue(0 <= s1 < env.state_space.n)
                    self.assertTrue(0 <= o < env.observation_space.n)
                    self.assertIn(r, env.rewards_dict.keys())
                    self.assertTrue(info is None or isinstance(info, dict))

            for a in range(env.action_space.n):
                with self.assertRaises(ValueError):
                    env.step_functional(-1, a)

            s1, o, r, done, info = env.step_functional(-1, -1)
            self.assertIsInstance(s1, int)
            self.assertIsInstance(o, int)
            self.assertIsInstance(r, float)
            self.assertIsInstance(done, bool)
            self.assertEqual((s1, o, r, done), (-1, -1, 0.0, True))
            self.assertTrue(info is None or isinstance(info, dict))

    def test_run(self):
        env = gym.make('POMDP-shopping_2-continuing-v0')
        self._test_run(env)
        env = gym.make('POMDP-shopping_2-episodic-v0')
        self._test_run(env)
        env = gym.make('POMDP-arrowtrail-continuing-v0')
        self._test_run(env)

    def _test_run(self, env):
        for _ in range(10):
            done = False
            env.reset()
            for _ in range(100):
                if done:
                    a = -1
                    o, r, done, info = env.step(a)
                    self.assertIsInstance(o, int)
                    self.assertIsInstance(r, float)
                    self.assertIsInstance(done, bool)
                    self.assertEqual((o, r, done), (-1, 0.0, True))
                    self.assertTrue(info is None or isinstance(info, dict))
                else:
                    a = env.action_space.sample()
                    o, r, done, info = env.step(a)
                    self.assertIsInstance(o, int)
                    self.assertIsInstance(r, float)
                    self.assertIsInstance(done, bool)
                    self.assertTrue(0 <= o < env.observation_space.n)
                    self.assertIn(r, env.rewards_dict.keys())
                    self.assertTrue(info is None or isinstance(info, dict))

    def test_seed(self):
        env = gym.make('POMDP-tiger-continuing-v0')
        actions = list(range(env.action_space.n)) * 20

        env.seed(17)
        env.reset()
        outputs = list(map(env.step, actions))

        # same seed
        env.seed(17)
        env.reset()
        outputs2 = list(map(env.step, actions))
        self.assertEqual(outputs, outputs2)

        # different seed
        env.seed(18)
        env.reset()
        outputs2 = list(map(env.step, actions))
        self.assertNotEqual(outputs, outputs2)


if __name__ == '__main__':
    unittest.main()
