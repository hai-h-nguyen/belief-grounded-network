import gym

import gym_pomdps

env = gym.make('POMDP-tag_avoid-episodic-v0')

s = env.reset_functional()

b = gym_pomdps.belief.belief_init(env)
a = 0
s, o, r, done, info = env.step_functional(a)
b = gym_pomdps.belief.belief_step(env, b, a, o)

print(b.shape)