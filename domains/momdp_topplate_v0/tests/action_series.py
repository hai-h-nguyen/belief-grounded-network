#!/usr/bin/env python

#
# Scanning th momdp_plates_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#
import time

import matplotlib.pyplot as plt

import gym

plt.style.use('ggplot')

env = gym.make(id='MomdpPlates-v0', rendering=True)
env.reset()
env.render()
print('Original z_g:', env.z_g)
print('Original theta:', env.angle_str[env.theta])
print("Original z_t:", env.z_t)
print()

actions = [1, 1, 0, 1]
z_g_all = []
angle_all = []
z_angle_dict = {}

for action in actions:
    env.render()
    state, reward, done, _ = env.step(action)
    # time.sleep(3)
    print("state:", state)
    print("Reward,", reward)
    print("Done", done)
    print('z_g:', env.z_g)
    print('theta:', env.angle_str[env.theta])
    print(env.belief)
    print()
    z_g_all.append(env.z_g)
    angle_all.append(env.theta)
    z_angle_dict[(env.action_str[action], env.z_g)] = env.angle_str[env.theta]

env.close()

print(z_g_all)
print(z_angle_dict)

plt.figure(figsize=(12, 8))
plt.xlabel('Angle')
plt.ylabel('z_g')
plt.plot(angle_all, z_g_all)
plt.show()
