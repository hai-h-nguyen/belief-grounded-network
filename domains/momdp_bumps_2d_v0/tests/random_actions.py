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

env = gym.make(id='MomdpBumps2d-v0', rendering=False)
belief = None

for i in range(10000):
    env.reset()
    env.render()
    belief = env.belief
    #time.sleep(0.2)

    print('Original x_g:', env.x_g)
    print('Original theta:', env.angle_str[env.theta])
    print("Bump #1 position:", env.x_bump1, env.y_bump1)
    print("Bump #2 position:", env.x_bump2, env.y_bump2)
    print(belief)

    done = False
    while not done:
        action = env.action_space.sample()
        print(action)
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('x_g:', env.x_g)
        print('y_g:', env.y_g)
        print('theta:', env.angle_str[env.theta])
        print(belief)
        print()

env.close()
