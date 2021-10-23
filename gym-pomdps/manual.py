#!/usr/bin/env python

import argparse
import shutil

import gym
import gym_pomdps


def manual_control(env, *, symbolic):
    """Manual control of environment.

    :param env:  Gym-POMDP environment
    :param symbolic:  Use non-semantic actions and observations
    """

    if symbolic:
        actions = [str(a) for a in range(len(env.model.actions))]
        observations = [str(o) for o in range(len(env.model.observations))]
    else:
        actions = env.model.actions
        observations = env.model.observations

    while True:
        env.reset()
        print('#' * shutil.get_terminal_size().columns)
        print('## START')
        while True:
            print(f'## Available actions:', ', '.join(actions))

            while True:  # select valid action
                try:
                    a = actions.index(input('>> '))
                except ValueError:
                    pass
                else:
                    break

            o, r, done, info = env.step(a)

            o = observations[o]
            print(f'## o = {o}')
            print(f'## r = {r}')
            if info:
                print(f'## info = {info}')

            if done:
                print('## DONE!')
                break


def main():
    parser = argparse.ArgumentParser('Manual Control')
    parser.add_argument('pomdp', choices=gym_pomdps.env_list)
    parser.add_argument('--symbolic', action='store_true')
    pargs = parser.parse_args()

    env = gym.make(pargs.pomdp)
    manual_control(env, symbolic=pargs.symbolic)


if __name__ == '__main__':
    main()
