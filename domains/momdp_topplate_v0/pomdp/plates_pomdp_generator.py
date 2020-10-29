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
    return f'g{state.g_pos.value}_theta{state.theta.value}_plate{state.p_pos.value}'


def all_states_fmt(all_states):
    """
    Formats all of the states given.
    """

    states_str_list = []
    first_states_str = state_fmt(all_states[0])
    g1 = first_states_str[1]
    g2 = first_states_str[2]

    for s in all_states:
        state_str = state_fmt(s)

        if state_str[1] != g1 or state_str[2] != g2:
            states_str_list.append('\n       ')
            g1 = state_str[1]
            g2 = state_str[2]

        states_str_list.append(state_str)

    return " ".join(states_str_list)


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


def all_obs_fmt(all_obs):
    """
    Formats all of the observations given.
    """

    obs_str_list = []
    first_obs_str = obs_fmt(all_obs[0])
    g1 = first_obs_str[1]
    g2 = first_obs_str[2]

    for s in all_obs:
        obs_str = obs_fmt(s)

        if obs_str[1] != g1 or obs_str[2] != g2:
            obs_str_list.append('\n             ')
            g1 = obs_str[1]
            g2 = obs_str[2]

        obs_str_list.append(obs_str)

    return " ".join(obs_str_list)


def is_start_state(state):
    """
    Checks if the given state is a start state.
    """
    return state.g_pos.value == 0 and state.theta.value == 'N'


def is_state_valid(state):
    """
    Checks if the given state is a valid state.
    """

    z_g = state.g_pos.value
    theta = state.theta.value
    z_t = state.p_pos.value

    if z_g == 0 or z_g > z_t:
        return theta == 'N'
    else:
        return theta != 'N'


def is_obs_valid(obs, max_plate_z):
    """
    Checks if the given observation is a valid observation.
    """

    z_g = obs.g_pos.value
    theta = obs.theta.value

    if z_g == 0 or z_g > max_plate_z:
        return theta == 'N'
    elif z_g == 1:
        return theta != 'N'
    else:
        return True


def generate_pomdp(g_pos=7, num_plate=4, gamma=0.95):
    assert (g_pos > num_plate)
    assert (num_plate > 0)
    assert (0 < gamma <= 1)

    # SPACES
    angle = ['D', 'N', 'U']
    angle_space = indextools.DomainSpace(angle)

    state_space = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(g_pos), theta=angle_space,
        p_pos=indextools.RangeSpace(1, num_plate + 1)  # Plate starts from index 1 to num_plate inclusively.
    )
    start_states = [
        s
        for s in state_space.elems
        if is_start_state(s)
    ]
    states = [
        s
        for s in state_space.elems
        if is_state_valid(s)
    ]

    action_list = ['U', 'D', 'G']
    action_space = indextools.DomainSpace(action_list)
    actions = list(action_space.elems)

    obs_space = indextools.JointNamedSpace(
        g_pos=indextools.RangeSpace(g_pos), theta=angle_space
    )
    observations = [
        o
        for o in obs_space.elems
        if is_obs_valid(o, num_plate)
    ]

    # PREAMBLE
    pomdp_strs = []

    pomdp_strs.append(f'# Plates Env POMDP\n')
    pomdp_strs.append(f'# This specific file was generated with the following parameters:\n')
    pomdp_strs.append(f'# g_pos: {g_pos}\n')
    pomdp_strs.append(f'# num_plate: {num_plate}\n')
    pomdp_strs.append(f'# gamma: {gamma}\n')
    pomdp_strs.append(f'# angle: {angle}\n')
    pomdp_strs.append(f'# Number of all actions: {len(actions)}\n')
    pomdp_strs.append(f'# Number of all states: {len(states)}\n')
    pomdp_strs.append(f'# Number of all observations: {len(observations)}\n')
    header_end_idx = len(pomdp_strs)  # Inserting position for the comments generated later.
    pomdp_strs.append('\n')

    pomdp_strs.append(f'discount: {gamma}\n')
    pomdp_strs.append('values: reward\n')
    pomdp_strs.append(f'states: {all_states_fmt(states)}\n')
    pomdp_strs.append(f'actions: {" ".join(action_fmt(a) for a in actions)}\n')
    pomdp_strs.append(f'observations: {all_obs_fmt(observations)}\n')

    # START
    pomdp_strs.append('\n')
    pomdp_strs.append(f'start include: {" ".join(state_fmt(s) for s in start_states)}\n')

    # TRANSITIONS
    pomdp_strs.append('\n')
    block_start_idx = len(pomdp_strs)  # For counting the state transition probabilities.
    for a in actions:
        # If grasps then resets.
        if a.value == 'G':
            pomdp_strs.append(f'T: {action_fmt(a)} : * reset\n')
            continue

        for s in states:
            if a.value == 'U':
                # Can still move up.
                if s.g_pos.value < g_pos - 1:
                    s_next = copy(s)
                    s_next.g_pos.value += 1

                    # s_next.g_pos between [1, the top plate].
                    if s_next.g_pos.value <= s.p_pos.value:
                        s_next.theta.value = 'D'

                    # s_next.g_pos > the top plate.
                    else:
                        s_next.theta.value = 'N'

                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} : {state_fmt(s_next)} 1.0\n')

                # Cannot move up then resets.
                else:
                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} reset\n')

            elif a.value == 'D':
                # Can still move down.
                if s.g_pos.value > 0:
                    s_next = copy(s)
                    s_next.g_pos.value -= 1

                    # At s_next.g_pos = 0, the angle is neutral.
                    if s_next.g_pos.value == 0:
                        s_next.theta.value = 'N'

                    # s_next.g_pos between [1, the top plate].
                    elif s_next.g_pos.value <= s.p_pos.value:
                        s_next.theta.value = 'U'

                    # s_next.g_pos > the top plate.
                    else:
                        s_next.theta.value = 'N'

                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} : {state_fmt(s_next)} 1.0\n')

                # Cannot move down then resets.
                else:
                    pomdp_strs.append(f'T: {action_fmt(a)} : {state_fmt(s)} reset\n')

    block_size = len(pomdp_strs) - block_start_idx
    pomdp_strs.insert(header_end_idx, f'# Number of state transition probabilities: {block_size}\n')

    # OBSERVATIONS
    pomdp_strs.append('\n')
    block_start_idx = len(pomdp_strs)  # For counting the observation probabilities.
    pomdp_strs.append("O: * : * : * 0.0\n")
    for a in actions:
        for s in states:
            for o in observations:
                if o.g_pos.value == s.g_pos.value and o.theta.value == s.theta.value:
                    pomdp_strs.append(f'O: {action_fmt(a)} : {state_fmt(s)} : {obs_fmt(o)} 1.0\n')

    header_end_idx += 1
    block_size = len(pomdp_strs) - block_start_idx
    pomdp_strs.insert(header_end_idx, f'# Number of observation probabilities: {block_size}\n')

    # REWARDS
    pomdp_strs.append('\n')
    block_start_idx = len(pomdp_strs)  # For counting the immediate rewards.
    for a in actions:
        if a.value == 'G':
            for s in states:
                if s.g_pos.value == s.p_pos.value:
                    pomdp_strs.append(f'R: {action_fmt(a)} : {state_fmt(s)} : * : * 1.0\n')

    header_end_idx += 1
    block_size = len(pomdp_strs) - block_start_idx
    pomdp_strs.insert(header_end_idx, f'# Number of immediate rewards: {block_size}\n')

    # Print.
    pomdp_text = "".join(pomdp_strs)
    print(pomdp_text)

    # Writes to a pomdp file.
    with open('../../../pomdps/plates.pomdp', 'w') as f:
        f.write(pomdp_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plates')
    parser.add_argument('--g-pos', type=int, default=7)
    parser.add_argument('--plate', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    generate_pomdp(config.g_pos, config.plate, config.gamma)
