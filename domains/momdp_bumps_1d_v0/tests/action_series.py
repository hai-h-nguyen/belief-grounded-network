#!/usr/bin/env python

#
# Scanning th momdp_plate_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#
import time

import matplotlib.pyplot as plt

import gym

plt.style.use('ggplot')

env = gym.make(id='MomdpBumps1dEnv-v0', rendering=True)
env.reset()
env.render()
print('Original x_g:', env.x_g)
print('Original theta:', env.angle_str[env.theta])
print("Bump #1 position:", env.x_bump1)
print("Bump #2 position:", env.x_bump2)
print()

actions = [2, 2, 3]
x_g_all = []
angle_all = []
x_angle_dict = {}

for action in actions:
    env.render()
    state, reward, done, _ = env.step(action)
    # time.sleep(3)
    print("state:", state)
    print("Reward,", reward)
    print("Done", done)
    print('x_g:', env.x_g)
    print('theta:', env.angle_str[env.theta])
    print(env.belief)
    print()
    x_g_all.append(env.x_g)
    angle_all.append(env.theta)
    x_angle_dict[(env.action_str[action], env.x_g)] = env.angle_str[env.theta]

env.close()

print(x_g_all)
print(x_angle_dict)

plt.figure(figsize=(12, 8))
plt.xlabel('Angle')
plt.ylabel('x_g')
plt.plot(angle_all, x_g_all)
plt.show()
