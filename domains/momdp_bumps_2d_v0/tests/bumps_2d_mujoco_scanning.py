#!/usr/bin/env python

#
# Scanning th momdp_plates_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#

import matplotlib.pyplot as plt
import gym

plt.style.use('ggplot')

env = gym.make(id='MomdpBumps2d-v0', rendering=False)
env.reset()
belief = None
x_g_all = []
angle_all = []
x_angle_dict = {}

while env.y_g < env.xy_g_high_limit:
    env.render()
    belief = env.belief

    print('Original gripah position:', env.x_g, env.y_g)
    print('Original theta:', env.angle_str[env.theta])
    print("Bump #1 position:", env.x_bump1, env.y_bump1)
    print("Bump #2 position:", env.x_bump2, env.y_bump2)
    print(belief)

    done = False
    while env.x_g < env.xy_g_high_limit and not done:
        action = 1
        env.render()
        _, _, done, _ = env.step(action)
        # belief = env.belief
        print('x_g:', env.x_g)
        print('y_g:', env.y_g)
        print('theta:', env.angle_str[env.theta])
        # print(belief)
        print()
        x_g_all.append(env.x_g)
        angle_all.append(env.theta)
        x_angle_dict[(env.action_str[action], env.x_g)] = env.angle_str[env.theta]

    done = False
    while env.x_g > env.xy_g_low_limit and not done:
        action = 0
        env.render()
        _, _, done, _ = env.step(action)
        belief = env.belief
        print('x_g:', env.x_g)
        print('y_g:', env.y_g)
        print('theta:', env.angle_str[env.theta])
        print(belief)
        print()
        x_g_all.append(env.x_g)
        angle_all.append(env.theta)
        x_angle_dict[(env.action_str[action], env.x_g)] = env.angle_str[env.theta]

    env.render()
    env.step(3)

print(x_g_all)
print(x_angle_dict)

for _x_bump_a in range(belief.shape[0]):
    for _y_bump_a in range(belief.shape[1]):
        for _x_bump_b in range(belief.shape[2]):
            for _y_bump_b in range(belief.shape[3]):
                if belief[_x_bump_a, _y_bump_a, _x_bump_b, _y_bump_b] == 0.5:
                    assert ((_x_bump_a == env.x_bump1 and _y_bump_a == env.y_bump1) or
                            (_x_bump_a == env.x_bump2 and _y_bump_a == env.y_bump2) or
                            (_x_bump_b == env.x_bump1 and _y_bump_b == env.y_bump1) or
                            (_x_bump_b == env.x_bump2 and _y_bump_b == env.y_bump2))

plt.figure(figsize=(12, 8))
plt.xlabel('Angle')
plt.ylabel('x_g')
plt.plot(angle_all, x_g_all)
plt.show()

env.close()
