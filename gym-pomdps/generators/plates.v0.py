#!/usr/bin/env python
import argparse
import math
from copy import copy

import indextools


def sfmt(s):
    pos_str = f'g{s.pos.g_pos.value}_theta{s.pos.theta.value}_plate{s.pos.p_pos.value}'
    return f'{pos_str}'


def afmt(a):
    return a.value

def ofmt(o):
    pos_str = f'g{o.pos.g_pos.value}_theta{o.pos.theta.value}'
    return f'{pos_str}'

def check_valid(s):
    z_g = s.pos.g_pos.value
    theta = s.pos.theta.value
    z_t = s.pos.p_pos.value

    cond1 = True
    cond2 = True

    if z_g >= z_t + 1 or z_g == 0:
        cond1 = (theta == 'N')
    else:
        cond2 = (theta != 'N')

    return (cond1 and cond2)    


def main():
    parser = argparse.ArgumentParser(description='Plate')
    parser.add_argument('--g-pos', type=int, default=6)
    parser.add_argument('--plate', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    angle = ['D', 'N', 'U']
    angle_space = indextools.DomainSpace(angle)    

    # Plate starts from index 1
    pos_space = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(config.g_pos), theta=angle_space, 
        p_pos=indextools.RangeSpace(1, config.plate)
    )

    state_space = indextools.JointNamedSpace(pos=pos_space)

    start_states = [
        s
        for s in state_space.elems
        if s.pos.g_pos.value == 0 and s.pos.theta.value == 'N' and check_valid(s)
    ]

    actions = ['U', 'D', 'G']
    action_space = indextools.DomainSpace(actions)

    obs = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(config.g_pos), theta=angle_space
    )    
    obs_space = indextools.JointNamedSpace(pos=obs)

    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {" ".join(sfmt(s) for s in state_space.elems if check_valid(s))}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    # START
    print()
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}')

    # TRANSITIONS
    print()

    for a in action_space.elems:
        if a.value == 'U':
            for s in state_space.elems:
                if check_valid(s):
                    # Can still move up
                    if s.pos.g_pos.value < config.g_pos - 1:
                        s1 = copy(s)
                        s1.pos.g_pos.value += 1

                        # Below the top plate
                        if s.pos.g_pos.value < s.pos.p_pos.value:
                            if s.pos.theta.value == 'N' or s.pos.theta.value == 'D':
                                s1.pos.theta.value = 'D'

                            else:
                                s1.pos.theta.value = 'U'

                        # The gripper is at least at the plate's position
                        else:
                            s1.pos.theta.value = 'N'

                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    # Cannot move up then reset
                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')

        elif a.value == 'D':
            for s in state_space.elems:
                # Check if s is a valid state
                if check_valid(s):
                    # Can still move down
                    if s.pos.g_pos.value > 0:
                        s1 = copy(s)
                        s1.pos.g_pos.value -= 1

                        # Below the top plate
                        # At 1, move down will zero the angle
                        if s.pos.g_pos.value == 1:
                            s1.pos.theta.value = 'N'

                        elif s.pos.g_pos.value <= s.pos.p_pos.value:
                            assert (s.pos.theta.value != 'N')
                            s1.pos.theta.value = s.pos.theta.value

                        # The gripper at the plate's position
                        elif s.pos.g_pos.value == s.pos.p_pos.value + 1:
                            assert(s.pos.theta.value == 'N')
                            s1.pos.theta.value = 'U'

                        # The gripper is at least 1 unit far from the plate's position
                        else:
                            s1.pos.theta.value = 'N'

                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    # Cannot move up then reset
                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')

        elif a.value == 'G':
            print(f'T: {afmt(a)}: * reset')

    # OBSERVATIONS
    print()
    print("O: * : * : * 0.0")
    for a in action_space.elems:
        for s in state_space.elems:
            if check_valid(s):
                for o in obs_space.elems:
                    if (o.pos.g_pos.value == s.pos.g_pos.value) and (o.pos.theta.value == s.pos.theta.value):
                        print(f'O: {afmt(a)}: {sfmt(s)}: {ofmt(o)} 1.0')

    # REWARDS
    print()
    for a in action_space.elems:

        if a.value == 'G':
            for s in state_space.elems:
                if s.pos.g_pos.value == s.pos.p_pos.value and check_valid(s):
                    print(f'R: {afmt(a)}: {sfmt(s)}: *: * 1.0')

if __name__ == '__main__':
    main()
