#!/usr/bin/env python

#
# Scanning th momdp_plates_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#

import matplotlib.pyplot as plt
import gym

plt.style.use('ggplot')

env = gym.make(id='MomdpPlates-v0', rendering=False)
belief = None

for i in range(10):
    env.reset()
    env.render()
    belief = env.belief

    print('Original z_g:', env.z_g)
    print('Original theta:', env.angle_str[env.theta])
    print("Original z_t:", env.z_t)
    print(belief)

    z_g_all = [env.z_g]
    theta_all = [env.theta]
    z_theta_dict = {}

    done = False
    while env.z_g < env.z_g_up_limit and not done:
        action = 1
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('z_g:', env.z_g)
        print('theta:', env.angle_str[env.theta])
        print(belief)
        print()
        z_g_all.append(env.z_g)
        theta_all.append(env.theta)
        z_theta_dict[(env.action_str[action], env.z_g)] = env.angle_str[env.theta]

    done = False
    while env.z_g > env.z_g_down_limit and not done:
        action = 0
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('z_g:', env.z_g)
        print('theta:', env.angle_str[env.theta])
        print(belief)
        print()
        z_g_all.append(env.z_g)
        theta_all.append(env.theta)
        z_theta_dict[(env.action_str[action], env.z_g)] = env.angle_str[env.theta]

    print(z_g_all)
    print(theta_all)
    print(z_theta_dict)
    print()

    for ii in range(belief.shape[0]):
        if belief[ii] == 1:
            assert (ii == env.z_t)

    plt.figure(figsize=(12, 8))
    plt.xlabel('Angle')
    plt.ylabel('z_g')
    plt.plot(theta_all, z_g_all)
    plt.show()

env.close()
