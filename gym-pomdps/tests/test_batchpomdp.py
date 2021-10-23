import unittest

import numpy as np
import numpy.random as rnd

import gym
import gym_pomdps


class Gym_BatchPOMDP_Test(unittest.TestCase):
    def test_functional(self):
        env = gym.make('POMDP-shopping_2-continuing-v0')
        self._test_functional(env, 5)
        env = gym.make('POMDP-shopping_2-episodic-v0')
        self._test_functional(env, 5)

    def _test_functional(self, env, batch_size):
        env = gym_pomdps.BatchPOMDP(env, batch_size)
        for _ in range(20):
            s = env.reset_functional()
            self.assertIsInstance(s, np.ndarray)
            self.assertEqual(s.dtype, int)
            self.assertTrue(((s >= 0) & (s < env.state_space.n)).all())

            s = rnd.randint(env.state_space.n, size=batch_size)
            a = rnd.randint(env.action_space.n, size=batch_size)

            s1, o, r, done, info = env.step_functional(s, a)
            self.assertIsInstance(s1, np.ndarray)
            self.assertEqual(s1.dtype, int)
            self.assertIsInstance(o, np.ndarray)
            self.assertEqual(o.dtype, int)
            self.assertIsInstance(r, np.ndarray)
            self.assertEqual(r.dtype, float)
            self.assertIsInstance(done, np.ndarray)
            self.assertEqual(done.dtype, bool)

            self.assertTrue((s1[done] == -1).all())
            self.assertTrue(
                ((s1[~done] >= 0) & (s1[~done] < env.state_space.n)).all()
            )
            self.assertTrue(((o >= 0) & (o < env.observation_space.n)).all())
            self.assertTrue(set(r).issubset(env.rewards_dict.keys()))
            self.assertTrue(info is None or isinstance(info, dict))

            s = np.full([batch_size], -1)
            a = rnd.randint(env.action_space.n, size=batch_size)
            with self.assertRaises(ValueError):
                env.step_functional(s, a)

            s = np.full([batch_size], -1)
            a = np.full([batch_size], -1)
            s1, o, r, done, info = env.step_functional(s, a)
            self.assertIsInstance(s1, np.ndarray)
            self.assertEqual(s1.dtype, int)
            self.assertIsInstance(o, np.ndarray)
            self.assertEqual(o.dtype, int)
            self.assertIsInstance(r, np.ndarray)
            self.assertEqual(r.dtype, float)
            self.assertIsInstance(done, np.ndarray)
            self.assertEqual(done.dtype, bool)
            self.assertTrue(info is None or isinstance(info, dict))
            self.assertTrue((s1 == -1).all())
            self.assertTrue((o == -1).all())
            self.assertTrue((r == 0.0).all())
            self.assertTrue(done.all())

    def test_run(self):
        env = gym.make('POMDP-shopping_2-continuing-v0')
        self._test_run(env, 5)
        env = gym.make('POMDP-shopping_2-episodic-v0')
        self._test_run(env, 5)

    def _test_run(self, env, batch_size):
        env = gym_pomdps.BatchPOMDP(env, batch_size)
        for _ in range(20):
            done = np.full(batch_size, False)
            env.reset()
            for _ in range(100):
                a = rnd.randint(env.action_space.n, size=batch_size)
                a[done] = -1
                o, r, done1, info = env.step(a)

                self.assertIsInstance(o, np.ndarray)
                self.assertIsInstance(r, np.ndarray)
                self.assertIsInstance(done1, np.ndarray)
                self.assertEqual(o.dtype, int)
                self.assertEqual(r.dtype, float)
                self.assertEqual(done1.dtype, bool)
                self.assertTrue(info is None or isinstance(info, dict))

                self.assertTrue((o[done & done1] == -1).all())
                self.assertTrue((r[done & done1] == 0.0).all())
                self.assertTrue((r[done & done1] == 0.0).all())
                self.assertTrue(done1[done].all())
                self.assertTrue(
                    (
                        (o[~done & done1] >= 0)
                        & (o[~done & done1] <= env.observation_space.n)
                    ).all()
                )
                self.assertTrue(
                    set(r[~done & done1]).issubset(env.rewards_dict.keys())
                )
                done = done1

    def test_seed(self):
        batch_size, num_steps = 20, 100

        env = gym.make('POMDP-tiger-continuing-v0')
        env = gym_pomdps.BatchPOMDP(env, batch_size)
        actions = rnd.randint(env.action_space.n, size=(num_steps, batch_size))

        # run environment multiple times with same seed
        env.seed(17)
        env.reset()
        output1 = list(map(env.step, actions))

        env.seed(17)
        env.reset()
        output2 = list(map(env.step, actions))

        for (o1, r1, done1, info1), (o2, r2, done2, info2) in zip(
            output1, output2
        ):
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(done1, done2)
            np.testing.assert_equal(info1, info2)

        env.seed(18)
        env.reset()
        output3 = list(map(env.step, actions))

        for (o1, r1, done1, info1), (o3, r3, done3, info3) in zip(
            output1, output3
        ):
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(o1, o3)
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(r1, r3)
            # in tiger, same actions means also same done!
            # with self.assertRaises(AssertionError):
            #     np.testing.assert_array_equal(done1, done3)
            with self.assertRaises(AssertionError):
                np.testing.assert_equal(info1, info3)

    def test_consistency(self):
        batch_size = 1  # single sample necessary to control randomness
        num_steps = 10

        env = gym.make('POMDP-loadunload-continuing-v0')
        actions = rnd.randint(env.action_space.n, size=num_steps)

        env.seed(17)
        env.reset()
        outputs1 = list(map(env.step, actions))

        env = gym_pomdps.BatchPOMDP(env, batch_size)
        actions = actions.reshape(-1, 1)

        env.seed(17)
        env.reset()
        outputs2 = list(map(env.step, actions))

        outputs = zip(outputs1, outputs2)
        for (o1, r1, done1, info1), (o2, r2, done2, info2) in outputs:
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(done1, done2)
            np.testing.assert_equal(info1, info2)


if __name__ == '__main__':
    unittest.main()
