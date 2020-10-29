#!/usr/bin/env python

#
# Scanning th momdp_plate_v0 in both the MuJoCo environment and the script environment.
#
# Created by Xinchao Song on June 1, 2020.
#

from multiprocessing import Process
import gym


def run(idx):
    print("Process #%d starts." % idx)

    env = gym.make(id='MomdpPlates-v0', rendering=False)
    env.reset()

    done = False
    while env.z_g < env.z_g_up_limit and not done:
        action = 2
        env.render()
        _, _, done, _ = env.step(action)

    done = False
    while env.z_g > env.z_g_down_limit and not done:
        action = 0
        env.render()
        _, _, done, _ = env.step(action)

    env.close()

    print("Process #%d ends." % idx)


task1 = Process(target=run, args=(1,))
task2 = Process(target=run, args=(2,))
task3 = Process(target=run, args=(3,))

task1.start()
task2.start()
task3.start()
