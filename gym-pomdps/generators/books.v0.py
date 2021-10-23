#!/usr/bin/env python

# POMDP File Format: http://www.pomdp.org/code/pomdp-file-spec.html
# gym-pomdps: https://github.com/abaisero/gym-pomdps

import argparse
from copy import copy

import indextools  # https://github.com/abaisero/one-to-one


def state_fmt(state):
    """
    Formats the given state.
    """
    return f'g{state.g_pos.value}_theta{state.theta.value}_book{state.b_pos.value}'


def action_fmt(action):
    """
    Formats the given action.
    """
    return action.value


def obs_fmt(obs):
    """
    Formats the given observations.
    """
    return f'g{obs.g_pos.value}_theta{obs.theta.value}'


def is_start_state(state):
    """
    Checks if the given state is a start state.
    """
    return (state.g_pos.value == 0) and (state.theta.value == 'N')


def is_state_valid(state, max_book_x):
    """
    Checks if the given state is a valid state.
    """

    x_g = state.g_pos.value
    theta = state.theta.value
    x_t = state.b_pos.value

    if (x_g == 0) or (x_g > max_book_x):
        return theta == 'N'
    elif x_g == x_t:
        return (theta == 'XL') or (theta == 'XR')
    else:
        return (theta == 'L') or (theta == 'R')


def is_obs_valid(obs, max_book_x):
    """
    Checks if the given observation is a valid observation.
    """

    x_g = obs.g_pos.value
    theta = obs.theta.value

    if (x_g == 0) or (x_g > max_book_x):
        return theta == 'N'
    else:
        return theta != 'N'


def main():
    parser = argparse.ArgumentParser(description='Plate')
    parser.add_argument('--g-pos', type=int, default=20)
    parser.add_argument('--theta', type=int, default=3)
    parser.add_argument('--book', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    assert (config.g_pos > config.book)
    assert (config.theta > 0)
    assert (config.book > 0)
    assert (0 < config.gamma <= 1)

    # SPACES
    angle = ['XL', 'L', 'N', 'R', 'XR']
    angle_space = indextools.DomainSpace(angle)

    state_space = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(config.g_pos), theta=angle_space,
        b_pos=indextools.RangeSpace(1, config.book + 1)  # Plate starts from index 1 to config.book inclusively.
    )
    start_states = [
        s
        for s in state_space.elems
        if is_start_state(s)
    ]
    states = [
        s
        for s in state_space.elems
        if is_state_valid(s, config.book)
    ]

    action_list = ['Right', 'Left', 'Grasp']
    action_space = indextools.DomainSpace(action_list)
    actions = list(action_space.elems)

    obs_space = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(config.g_pos), theta=angle_space
    )
    observations = [
        o
        for o in obs_space.elems
        if is_obs_valid(o, config.book)
    ]

    # PREAMBLE
    pomdp_strs = []

    pomdp_strs.append(f'# This specific file was generated with parameters:\n')
    pomdp_strs.append(f'# {config}\n')
    pomdp_strs.append('\n')
    pomdp_strs.append(f'discount: {config.gamma}\n')
    pomdp_strs.append('values: reward\n')

    pomdp_strs.append(f'states: {" ".join(state_fmt(s) for s in states)}\n')
    pomdp_strs.append(f'actions: {" ".join(action_fmt(a) for a in actions)}\n')
    pomdp_strs.append(f'observations: {" ".join(obs_fmt(o) for o in observations)}\n')

    # START
    pomdp_strs.append('\n')
    pomdp_strs.append(f'start include: {" ".join(state_fmt(s) for s in start_states)}\n')

    # TRANSITIONS
    pomdp_strs.append('\n')
    for a in actions:
        # If grasps then resets.
        if a.value == 'Grasp':
            pomdp_strs.append(f'T: {action_fmt(a)} : * reset\n')
            continue

        for s in states:
            if a.value == 'Right':
                # Can still move right.
                if s.g_pos.value < config.g_pos - 1:
                    s1 = copy(s)
                    s1.g_pos.value += 1

                    # x_g between [1, the rightmost book].
                    if s1.g_pos.value <= config.book:
                        if s1.g_pos.value == s.b_pos.value:
                            s1.theta.value = 'XL'
                        else:
                            s1.theta.value = 'L'

                    # The gripper is at least at the book's position.
                    else:
                        s1.theta.value = 'N'

                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} : {state_fmt(s1)} 1.0\n')

                # Cannot move right then resets.
                else:
                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} reset\n')

            elif a.value == 'Left':
                # Can still move left.
                if s.g_pos.value > 0:
                    s1 = copy(s)
                    s1.g_pos.value -= 1

                    # At x_g = 0, the angle is neutral.
                    if s1.g_pos.value == 0:
                        s1.theta.value = 'N'

                    # x_g between [0, the rightmost book].
                    elif s1.g_pos.value <= config.book:
                        if s1.g_pos.value == s.b_pos.value:
                            s1.theta.value = 'XR'
                        else:
                            s1.theta.value = 'R'

                    # The gripper is at least 1 unit far from the book's position.
                    else:
                        s1.theta.value = 'N'

                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} : {state_fmt(s1)} 1.0\n')

                # Cannot move left then resets.
                else:
                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} reset\n')

    # OBSERVATIONS
    pomdp_strs.append('\n')
    pomdp_strs.append("O: * : * : * 0.0\n")
    for a in actions:
        for s in states:
            for o in observations:
                if (o.g_pos.value == s.g_pos.value) and (o.theta.value == s.theta.value):
                    pomdp_strs.append(f'O: {action_fmt(a)} : {state_fmt(s)} : {obs_fmt(o)} 1.0\n')

    # REWARDS
    pomdp_strs.append('\n')
    for a in actions:
        if a.value == 'Grasp':
            for s in states:
                if s.g_pos.value == s.b_pos.value:
                    pomdp_strs.append(f'R: {action_fmt(a)} : {state_fmt(s)} : * : * 1.0\n')

    pomdp_text = "".join(pomdp_strs)
    print(pomdp_text)

    # Writes to a pomdp file.
    with open('books.pomdp', 'w') as f:
        f.write(pomdp_text)


if __name__ == '__main__':
    main()
