#!/usr/bin/env python
import argparse
import math
from copy import copy

import indextools


def sfmt(s):
    return f'cart{s.pos.cart.value}_theta{s.pos.angle.value}_lb{s.pos.lbump.value}_rb{s.pos.rbump.value}'

def afmt(a):
    return a.value

def ofmt(o):
    pos_str = f'cart{o.pos.cart.value}_theta{o.pos.angle.value}'
    return f'{pos_str}'

def check_valid(s, min_bump_distance, max_bump_distance):
    cart = s.pos.cart.value
    theta = s.pos.angle.value
    lbump = s.pos.lbump.value
    rbump = s.pos.rbump.value

    # Conditions for checking
    cond1 = (rbump - lbump >= min_bump_distance)
    cond2 = (rbump - lbump <= max_bump_distance)

    cond3 = True
    if (cart == lbump or cart == rbump):
        cond3 = not(theta == 'Z')

    return (cond1 and cond2 and cond3)   


def main():
    parser = argparse.ArgumentParser(description='Plate')
    parser.add_argument('--cart', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    angle = ['L', 'Z', 'R']
    angle_space = indextools.DomainSpace(angle)

    cart_pos_min = 0
    cart_pos_max = config.cart

    min_bump_distance = 2
    max_bump_distance = int(cart_pos_max / 2) 

    pos_space = indextools.JointNamedSpace(
        cart=indextools.RangeSpace(cart_pos_min, cart_pos_max + 1), 
        angle=angle_space, 
        lbump=indextools.RangeSpace(cart_pos_min + 1, cart_pos_max), 
        rbump=indextools.RangeSpace(cart_pos_min + 1, cart_pos_max) 
    )

    check_state = lambda s: check_valid(s, min_bump_distance, max_bump_distance)

    state_space = indextools.JointNamedSpace(pos=pos_space)
    all_states = [
        s
        for s in state_space.elems
        if check_state(s)
    ]

    start_states = [
        s
        for s in state_space.elems
        if s.pos.angle.value == 'Z' and check_state(s)
    ]

    actions = ['LS', 'LH', 'RS', 'RH']
    action_space = indextools.DomainSpace(actions)

    obs = indextools.JointNamedSpace(
        cart=indextools.RangeSpace(cart_pos_min, cart_pos_max + 1), angle=angle_space
    )    
    obs_space = indextools.JointNamedSpace(pos=obs)
    all_obs = [
        s
        for s in obs_space.elems
    ]    

    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print(f'# observations: {len(all_obs)}')
    print(f'# actions: {len(actions)}')
    print(f'# states: {len(all_states)}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {" ".join(sfmt(s) for s in state_space.elems if check_state(s))}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    # START
    print()
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}')    

    # TRANSITIONS
    print()

    for a in action_space.elems:
        if a.value == 'LS':
            for s in state_space.elems:
                if check_state(s):
                    s1 = copy(s)
                    # Can still move left
                    if (s.pos.cart.value >= cart_pos_min + 1):
                        s1.pos.cart.value -= 1

                        # one of bumps is on the left
                        if ((s.pos.cart.value == s.pos.rbump.value + 1) or (s.pos.cart.value == s.pos.lbump.value + 1)):
                            s1.pos.angle.value = 'R'
                        else:
                            s1.pos.angle.value = 'Z'

                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')

        if a.value == 'RS':
            for s in state_space.elems:
                if check_state(s):
                    s1 = copy(s)
                    # Can still move right
                    if (s.pos.cart.value <= cart_pos_max - 1):
                        s1.pos.cart.value += 1

                        # one of bumps is on the right then the finger points LEFT
                        if ((s.pos.cart.value == s.pos.rbump.value - 1) or (s.pos.cart.value == s.pos.lbump.value - 1)):
                            s1.pos.angle.value = 'L'
                        else:
                            s1.pos.angle.value = 'Z'

                        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')

        if a.value == 'LH':
            for s in state_space.elems:
                if check_state(s):
                    s1 = copy(s)
                    # Can still move left
                    if (s.pos.cart.value >= cart_pos_min + 1):
                        s1.pos.cart.value -= 1

                        # left/right bump is here then it is moved left if angle is positive
                        if ((s.pos.cart.value in [s.pos.rbump.value, s.pos.lbump.value]) and (s.pos.angle.value == 'R')):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # left/right bump is on the left then it is moved left if angle is zero or point left
                        elif ((s.pos.cart.value - 1 in [s.pos.lbump.value, s.pos.rbump.value]) and (s.pos.angle.value in ['Z', 'L'])):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        else:
                            print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')

        if a.value == 'RH':
            for s in state_space.elems:
                if check_state(s):                
                    s1 = copy(s)
                    # Can still move right
                    if (s.pos.cart.value <= cart_pos_max - 1):
                        s1.pos.cart.value += 1

                        # left/right bump is here then it is moved left if finger points left
                        if ((s.pos.cart.value in [s.pos.rbump.value, s.pos.lbump.value]) and (s.pos.angle.value == 'L')):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # left/right bump is on the right then it is moved left if angle is zero or points right
                        elif ((s.pos.cart.value + 1 in [s.pos.lbump.value, s.pos.rbump.value]) and (s.pos.angle.value in ['Z', 'R'])):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        else:
                            print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')                    

    # OBSERVATIONS
    print()
    for a in action_space.elems:
        for s in state_space.elems:
            if check_state(s):
                for o in obs_space.elems:
                    if (o.pos.cart.value == s.pos.cart.value) and (o.pos.angle.value == s.pos.angle.value):
                        print(f'O: {afmt(a)}: {sfmt(s)}: {ofmt(o)} 1.0')

    # REWARDS
    print()
    for a in action_space.elems:
        if a.value == 'RH':
            for s in state_space.elems:
                if check_state(s):
                    if s.pos.cart.value == s.pos.rbump.value and s.pos.angle.value == 'L':
                        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 1.0')

                    if s.pos.cart.value + 1 == s.pos.rbump.value and s.pos.angle.value in ['Z', 'R']:
                        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 1.0')

if __name__ == '__main__':
    main()