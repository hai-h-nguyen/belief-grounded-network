import gym
import numpy as np

from gym_pomdps.envs import POMDP

__all__ = ['BatchPOMDP']


class BatchPOMDP(gym.Wrapper):
    """Simulates multiple POMDP trajectories at the same time."""

    def __init__(self, env, batch_size):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Env is not a POMDP (got {type(env)}).')
        if batch_size <= 0:
            raise ValueError(f'Batch size is not positive (got ({batch_size}).')

        super().__init__(env)
        self.batch_size = batch_size
        self.state = np.full([batch_size], -1)

    def reset(self):  # pylint: disable=arguments-differ
        self.state = self.reset_functional()

    def step(self, action):
        self.state, *ret = self.step_functional(self.state, action)
        return ret

    def reset_functional(self):
        if self.env.start is None:
            state = self.np_random.randint(
                self.state_space.n, size=self.batch_size
            )
        else:
            state = self.np_random.multinomial(
                1, self.env.start, size=self.batch_size
            ).argmax(1)
        return state

    def step_functional(self, state, action):
        if ((state == -1) != (action == -1)).any():
            raise ValueError(f'Invalid state-action pair ({state}, {action}).')

        shape = state.shape
        mask = state != -1

        state1 = np.full(shape, -1)
        obs = np.full(shape, -1)
        reward = np.full(shape, 0.0)
        reward_cat = np.full(shape, -1)
        done = np.full(shape, True)

        if mask.any():
            # unmasked states should be within bounds
            s, a = state[mask], action[mask]
            assert ((s >= 0) & (s < self.state_space.n)).all()
            assert ((a >= 0) & (a < self.action_space.n)).all()

            s1 = np.array(
                [
                    self.np_random.multinomial(1, p).argmax()
                    for p in self.env.T[s, a]
                ]
            )
            o = np.array(
                [
                    self.np_random.multinomial(1, p).argmax()
                    for p in self.env.O[s, a, s1]
                ]
            )
            # NOTE below is the same but unified in single sampling op; requires TO
            # s1, o = np.array([
            #     divmod(
            #         self.np_random.multinomial(1, p.ravel()).argmax(),
            #         self.observation_space.n,
            #     )
            #     for p in self.env.TO[s, a]
            # ]).T

            r = self.env.R[s, a, s1, o]

            if self.env.episodic:
                d = self.env.D[s, a]
                s1[d] = -1
            else:
                d = np.zeros(mask.sum(), dtype=bool)

            state1[mask] = s1
            obs[mask] = o
            reward[mask] = r
            done[mask] = d

            reward_cat[mask] = [self.rewards_dict[r_] for r_ in r]
            info = dict(reward_cat=reward_cat)

        info = dict(reward_cat=reward_cat)
        return state1, obs, reward, done, info
