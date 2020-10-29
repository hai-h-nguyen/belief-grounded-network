#!/usr/bin/env python

#
# Collects state data for action 0 & 1 from plates_serve_simple_v0_mujoco.
# Need to change "return np.array([self.x_g, theta_discretized])" to "return [self.x_g, round(theta, 2)]"
# in momdp_plates_mujoco.py before running.
#
# Created by Xinchao Song on June 14, 2020.
#

import gym

env = gym.make(id='PlatesServeSimpleMuJoCo-v0', recompile_cpp=True, rendering=False)
state = {}


def remember_and_print(_num_plate, _last_obs, _obs, _action):
    state[(_num_plate, _last_obs, _action)] = tuple(_obs)
    print((_num_plate, _last_obs, _action), state[(_num_plate, _last_obs, _action)])
    return tuple(_obs)


for num_plate in range(1, 5):
    env.num_plate_determined = num_plate

    for floor in range(env.x_g_right_limit + 1):
        for action in range(2):
            # 1
            obs = env.reset()
            last_obs = tuple(obs)

            while obs[0] < floor:
                obs, _, _, _ = env.step(0)
                last_obs = remember_and_print(num_plate, last_obs, obs, 0)

            done = False
            while not done:
                obs, _, done, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)

            # 2
            obs = env.reset()
            last_obs = tuple(obs)

            while obs[0] < floor:
                obs, _, _, _ = env.step(0)
                last_obs = remember_and_print(num_plate, last_obs, obs, 0)

            done = False
            while not done:
                obs, _, done, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)
                obs, _, _, _ = env.step(1 - action)
                last_obs = remember_and_print(num_plate, last_obs, obs, 1 - action)
                obs, _, _, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)

            # 3
            obs = env.reset()
            last_obs = tuple(obs)

            while obs[0] < env.x_g_right_limit:
                obs, _, _, _ = env.step(0)
                last_obs = remember_and_print(num_plate, last_obs, obs, 0)

            while obs[0] > floor:
                obs, _, _, _ = env.step(1)
                last_obs = remember_and_print(num_plate, last_obs, obs, 1)

            done = False
            while not done:
                obs, _, done, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)

            # 4
            obs = env.reset()
            last_obs = tuple(obs)

            while obs[0] < env.x_g_right_limit:
                obs, _, _, _ = env.step(0)
                last_obs = remember_and_print(num_plate, last_obs, obs, 0)

            while obs[0] > floor:
                obs, _, _, _ = env.step(1)
                last_obs = remember_and_print(num_plate, last_obs, obs, 1)

            done = False
            while not done:
                obs, _, done, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)
                obs, _, _, _ = env.step(1 - action)
                last_obs = remember_and_print(num_plate, last_obs, obs, 1 - action)
                obs, _, _, _ = env.step(action)
                last_obs = remember_and_print(num_plate, last_obs, obs, action)

env.close()

print(state)
