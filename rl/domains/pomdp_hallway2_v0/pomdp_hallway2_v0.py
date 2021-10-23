import gym
import gym_pomdps
from gym import spaces
import numpy as np

class PomdpHallway2V0(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.env = gym.make("POMDP-hallway2-episodic-v0")
		self.action_space = self.env.action_space
		self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.int32)
		self.state_space = self.env.state_space
		self.discount = self.env.discount

	def close(self):
		pass

	def seed(self, seed):
		self.env.seed(seed)

	def get_state(self):
		return self.state

	def get_belief(self):
		return self.belief

	def reset(self):
		self.state = self.env.reset_functional()
		self.belief = gym_pomdps.belief.belief_init(self.env)

		initial_action = np.random.randint(0, self.action_space.n)
		initial_obs, _, _, _ = self.step(initial_action)
		return initial_obs

	def step(self, action):
		if isinstance(action, np.ndarray):
			action = action[0]

		# Next state
		r = gym_pomdps.belief.expected_reward(self.env, self.belief, action)
		self.state, o, r, done, info = self.env.step_functional(self.state, action)

		# Update next belief
		self.belief = gym_pomdps.belief.belief_step(self.env, self.belief, action, o)

		info['belief'] = self.belief
		info['curr_state'] = np.array([self.state])

		return np.array([o, action]), r, done, info
