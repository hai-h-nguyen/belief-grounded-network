import gym
import numpy as np

from gym_pomdps.belief import belief_init, belief_step, expected_reward
from gym_pomdps.envs import POMDP

__all__ = ['BeliefMDP']


class BeliefMDP(gym.Wrapper):
    """Belief-MDP associated with input POMDP."""

    def __init__(self, env):
        if not isinstance(env.unwrapped, POMDP):
            raise TypeError(f'Env is not a POMDP (got {type(env)}).')

        super().__init__(env)
        self.belief = None

    def reset(self):  # pylint: disable=arguments-differ
        self.env.reset()
        self.belief = belief_init(self.env, np.asarray(self.state).shape)
        return self.belief

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info.update({'observation': observation, 'reward': reward})

        # NOTE:  compute reward before the step is made!
        expected_reward_ = expected_reward(self.env, self.belief, action)
        self.belief = belief_step(self.env, self.belief, action, observation)

        return self.belief, expected_reward_, done, info
