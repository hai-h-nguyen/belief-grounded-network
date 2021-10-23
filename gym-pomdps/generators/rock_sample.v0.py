#!/usr/bin/env python
import argparse
import math
from copy import copy

import indextools


def sfmt(s):
    pos_str = f's_{s.pos.x.value}_{s.pos.y.value}'
    rocks_str = ''.join([str(int(rock.value)) for rock in s.rocks])
    return f'{pos_str}_{rocks_str}'


def afmt(a):
    return a.value


def ofmt(o):
    return o.value


def main():
    parser = argparse.ArgumentParser(description='RockSample')
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    assert config.n > 1
    assert config.k > 0

    if config.n == 5 and config.k == 6:

        # #######
        # #  R  #
        # #R  R #
        # #A    #
        # # RR  #
        # #    R#
        # #######

        base, d0 = 2, 20
        rock_positions = [(0, 1), (1, 3), (2, 0), (2, 3), (3, 1), (4, 4)]
    elif config.n == 7 and config.k == 8:

        # #########
        # #  R    #
        # #R  R   #
        # #       #
        # #A     R#
        # #  RR   #
        # #     R #
        # # R     #
        # #########

        base, d0 = 2, 20
        rock_positions = [
            (0, 1),
            (1, 6),
            (2, 0),
            (2, 4),
            (3, 1),
            (3, 4),
            (5, 5),
            (6, 3),
        ]
    elif config.n == 11 and config.k == 11:

        # #############
        # #           #
        # #      R    #
        # #           #
        # #R  RR    R #
        # #  R        #
        # #A          #
        # #           #
        # #R          #
        # # R R R     #
        # #         R #
        # #           #
        # #############

        base, d0 = 8, 20
        rock_positions = [
            (0, 3),
            (0, 7),
            (1, 8),
            (2, 4),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ]
    else:
        raise ValueError(f'Invalid sizes (n={config.n}, k={config.k})')

    pos_space = indextools.JointNamedSpace(
        x=indextools.RangeSpace(config.n), y=indextools.RangeSpace(config.n)
    )

    rock_space = indextools.BoolSpace()
    rocks_space = indextools.JointSpace(*[rock_space] * config.k)

    state_space = indextools.JointNamedSpace(pos=pos_space, rocks=rocks_space)

    actions = ['N', 'S', 'E', 'W', 'sample'] + [
        f'check_{i}' for i in range(config.k)
    ]
    action_space = indextools.DomainSpace(actions)

    obs = ['none', 'good', 'bad']
    obs_space = indextools.DomainSpace(obs)

    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {" ".join(sfmt(s) for s in state_space.elems)}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    start_states = [
        s
        for s in state_space.elems
        if s.pos.x.value == 0 and s.pos.y.value == config.n // 2
    ]

    # START
    print()
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}')

    # TRANSITIONS
    print()
    for a in action_space.elems:
        print(f'T: {afmt(a)} identity')

        if a.value == 'N':
            for s in state_space.elems:
                if s.pos.y.value < config.n - 1:
                    s1 = copy(s)
                    s1.pos.y.value += 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'S':
            for s in state_space.elems:
                if s.pos.y.value > 0:
                    s1 = copy(s)
                    s1.pos.y.value -= 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'E':
            for s in state_space.elems:
                if s.pos.x.value == config.n - 1:
                    print(f'T: {afmt(a)}: {sfmt(s)} reset')
                else:
                    s1 = copy(s)
                    s1.pos.x.value += 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'W':
            for s in state_space.elems:
                if s.pos.x.value > 0:
                    s1 = copy(s)
                    s1.pos.x.value -= 1
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                    print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value == 'sample':
            for s in state_space.elems:
                try:
                    rock_i = rock_positions.index(
                        (s.pos.x.value, s.pos.y.value)
                    )
                except ValueError:
                    pass
                else:
                    if s.rocks[rock_i]:
                        s1 = copy(s)
                        s1.rocks[rock_i].value = False
                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s)} 0.0')
                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

        elif a.value.startswith('check_'):
            pass  # no state-transition

    # OBSERVATIONS
    print()
    print('O: *: *: none 1.0')
    for a in action_space.elems:
        if a.value.startswith('check_'):
            print(f'O: {afmt(a)}: *: none 0.0')
            for s1 in state_space.elems:
                rock_i = int(a.value[len('check_') :])
                rock_pos = rock_positions[rock_i]
                rock_good = bool(s1.rocks[rock_i].value)
                pos = s1.pos.x.value, s1.pos.y.value

                dist = math.sqrt(
                    (pos[0] - rock_pos[0]) ** 2 + (pos[1] - rock_pos[1]) ** 2
                )
                efficiency = base ** (-dist / d0)
                pcorrect = 0.5 * (1 + efficiency)
                pgood = pcorrect if rock_good else 1 - pcorrect

                print(f'O: {afmt(a)}: {sfmt(s1)}: good {pgood:.6f}')
                print(f'O: {afmt(a)}: {sfmt(s1)}: bad {1 - pgood:.6f}')

    # REWARDS
    print()
    for a in action_space.elems:

        if a.value == 'E':
            for s in state_space.elems:
                if s.pos.x.value == config.n - 1:
                    print(f'R: {afmt(a)}: {sfmt(s)}: *: * 10.0')

        elif a.value == 'sample':
            # TODO how to handle -100.0 actions, like bumping into a wall?
            print(f'R: {afmt(a)}: *: *: * -10.0')
            for s in state_space.elems:
                try:
                    rock_i = rock_positions.index(
                        (s.pos.x.value, s.pos.y.value)
                    )
                except ValueError:
                    pass
                else:
                    if s.rocks[rock_i].value:
                        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 10.0')


if __name__ == '__main__':
    main()
