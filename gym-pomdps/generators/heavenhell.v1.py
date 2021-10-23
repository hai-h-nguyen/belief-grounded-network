#!/usr/bin/env python
import argparse
from copy import copy

import indextools


def sfmt(s):
    return str(s.idx)


def afmt(a):
    return a.value


def ofmt(o):
    return o.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shopping')
    parser.add_argument('n', type=int, default=None)
    # parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    config = parser.parse_args()

    assert config.n >= 1
    assert 0 < config.gamma <= 1

    ncells = 2 + 4 * config.n
    cell_space = indextools.RangeSpace(ncells)
    heaven_space = indextools.DomainSpace(['left', 'right'])

    state_space = indextools.JointNamedSpace(
        heaven=heaven_space, cell=cell_space
    )

    actions = ['N', 'S', 'E', 'W']
    action_space = indextools.DomainSpace(actions)

    obs = [f'o{i}' for i in range(cell_space.nelems - 1)] + ['left', 'right']
    obs_space = indextools.DomainSpace(obs)

    print(
        """# A robot will be rewarded +1 for attaining heaven in one
# if it accidently reaches hell it will get -1
# Problem is attributed to Sebastian Thrun but first appeared in Geffner
# & Bonet: Solving Large POMDPs using Real Time DP 1998.
# A priest is available to tell it where heaven is (left or right)
#
#        Heaven  4  3  2  5  6  Hell
#                      1
#                      0
#                      7  8  9 Priest
#
#          Hell 14 13 12 15 16  Heaven
#                     11
#                     10
#                     17 18 19 Priest
#
# Furthermore, the map observations may be noisy. Edit the file to change
# the level of noise.
# Heaven is obtained by moving W in state 4 or E in 16 and hell is
# obtained by moving E in 6 and W in 14. The priest is in 9
# The agent starts at 0"""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {state_space.nelems}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    start_states = [s for s in state_space.elems if s.cell.value == 0]

    # START
    print()
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}')

    # mid_states = [
    #     s
    #     for s in state_space.elems
    #     if 0 <= s.cell.idx <= config.n
    #     or s.cell.idx == cell_space.nelems - config.n - 1
    # ]
    # top_left_states = [s for s in state_space.elems if True]
    # top_right_states = [s for s in state_space.elems if True]
    # bottom_states = [s for s in state_space.elems if True]

    # TRANSITIONS
    print()
    for a in action_space.elems:
        print(f'T: {afmt(a)} identity')
        if a.value == 'N':
            for s in state_space.elems:
                if 0 <= s.cell.value < config.n:
                    s1 = copy(s)
                    s1.cell.value += 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                elif s.cell.value == cell_space.nelems - config.n - 1:
                    s1 = copy(s)
                    s1.cell.value = 0
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'S':
            for s in state_space.elems:
                if 0 < s.cell.value <= config.n:
                    s1 = copy(s)
                    s1.cell.value -= 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                elif s.cell.value == 0:
                    s1 = copy(s)
                    s1.cell.value = cell_space.nelems - config.n - 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'E':
            for s in state_space.elems:
                if config.n + 1 <= s.cell.value < 2 * config.n:
                    s1 = copy(s)
                    s1.cell.value -= 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                if s.cell.value == config.n:
                    s1 = copy(s)
                    s1.cell.value = 2 * config.n + 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                if (
                    2 * config.n < s.cell.value < 3 * config.n
                    or 3 * config.n < s.cell.value < cell_space.nelems - 1
                ):
                    s1 = copy(s)
                    s1.cell.value += 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'W':
            for s in state_space.elems:
                if config.n <= s.cell.value < 2 * config.n:
                    s1 = copy(s)
                    s1.cell.value += 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                elif s.cell.value == 2 * config.n + 1:
                    s1 = copy(s)
                    s1.cell.value = config.n
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')
                elif (
                    2 * config.n < s.cell.value <= 3 * config.n
                    or 3 * config.n + 1 < s.cell.value < cell_space.nelems
                ):
                    s1 = copy(s)
                    s1.cell.value -= 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

    for s in state_space.elems:
        if s.cell.value == 2 * config.n or s.cell.value == 3 * config.n:
            print(f'T: *: {sfmt(s)} reset')

    # OBSERVATIONS
    print()
    for s1 in state_space.elems:
        if s1.cell.value < cell_space.nelems - 1:
            print(f'O: *: {sfmt(s1)}: o{s1.cell.value} 1.0')
        else:  # if s1.cell.value == cell_space.nelems - 1:
            print(f'O: *: {sfmt(s1)}: {s1.heaven.value} 1.0')

    # REWARDS
    print()
    for s in state_space.elems:
        if (
            s.cell.value == 2 * config.n
            and s.heaven.value == 'left'
            or s.cell.value == 3 * config.n
            and s.heaven.value == 'right'
        ):
            print(f'R: *: {sfmt(s)}: *: * 1.0')
        elif (
            s.cell.value == 2 * config.n
            and s.heaven.value == 'right'
            or s.cell.value == 3 * config.n
            and s.heaven.value == 'left'
        ):
            print(f'R: *: {sfmt(s)}: *: * -1.0')
