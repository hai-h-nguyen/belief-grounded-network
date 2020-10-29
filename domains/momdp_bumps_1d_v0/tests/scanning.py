#!/usr/bin/env python

#
# Scanning th momdp_plate_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#

import matplotlib.pyplot as plt
import gym

plt.style.use('ggplot')

env = gym.make(id='MomdpBumps1dEnv-v0', rendering=True)
belief = None

for i in range(10):
    env.reset()
    env.render()
    belief = env.belief

    print('Original x_g:', env.x_g)
    print('Original theta:', env.angle_str[env.theta])
    print("Bump #1 position:", env.x_bump1)
    print("Bump #2 position:", env.x_bump2)
    #print(belief)

    x_g_all = []
    angle_all = []
    x_angle_dict = {}

    done = False
    while env.x_g < env.x_g_right_limit and not done:
        action = 2
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('x_g:', env.x_g)
        print('theta:', env.angle_str[env.theta])
        #print(belief)
        print()
        x_g_all.append(env.x_g)
        angle_all.append(env.theta)
        x_angle_dict[(env.action_str[action], env.x_g)] = env.angle_str[env.theta]

    done = False
    while env.x_g > env.x_g_left_limit and not done:
        action = 0
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('x_g:', env.x_g)
        print('theta:', env.angle_str[env.theta])
        #print(belief)
        print()
        x_g_all.append(env.x_g)
        angle_all.append(env.theta)
        x_angle_dict[(env.action_str[action], env.x_g)] = env.angle_str[env.theta]

    print(x_g_all)
    print(x_angle_dict)

    for ii in range(belief.shape[0]):
        for jj in range(belief.shape[1]):
            if belief[ii, jj] == 1:
                assert (ii == env.x_bump1 and jj == env.x_bump2)

    plt.figure(figsize=(12, 8))
    plt.xlabel('Angle')
    plt.ylabel('x_g')
    plt.plot(angle_all, x_g_all)
    plt.show()

env.close()
