#!/usr/bin/env python
import argparse
import math
from copy import copy

import indextools


def sfmt(s):
    return f'cart{s.pos.cart.value}_theta{s.pos.angle.value}_lb{s.pos.sbump.value}_rb{s.pos.bbump.value}'

def afmt(a):
    return a.value

def ofmt(o):
    pos_str = f'cart{o.pos.cart.value}_theta{o.pos.angle.value}'
    return f'{pos_str}'

def check_valid(s, min_cart_pos, max_cart_pos, min_bump_pos, max_bump_pos, min_bump_distance):
    cart = s.pos.cart.value
    theta = s.pos.angle.value
    sbump = s.pos.sbump.value
    bbump = s.pos.bbump.value

    # Conditions for checking
    cond1 = abs(bbump - sbump) >= min_bump_distance
    cond2 = (sbump >= min_bump_pos) and (bbump >= min_bump_pos)
    cond3 = (sbump <= max_bump_pos) and (bbump <= max_bump_pos)
    cond4 = (cart >= min_cart_pos) and (cart <= max_cart_pos)

    return (cond1 and cond2 and cond3 and cond4)   


def main():
    parser = argparse.ArgumentParser(description='Plate')
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.95)
    config = parser.parse_args()

    angle = ['LB', 'LS', 'Z', 'RS', 'RB']
    angle_space = indextools.DomainSpace(angle)

    min_cart_pos = 0
    max_cart_pos = config.size - 1

    min_bump_distance = 2

    min_bump_pos = 1
    max_bump_pos = max_cart_pos - 1

    pos_space = indextools.JointNamedSpace(
        cart=indextools.RangeSpace(min_cart_pos, max_cart_pos + 1), 
        angle=angle_space, 
        sbump=indextools.RangeSpace(min_bump_pos, max_bump_pos + 1), # small bump
        bbump=indextools.RangeSpace(min_bump_pos, max_bump_pos + 1)  # big bump
    )

    check_state = lambda s: check_valid(s, min_cart_pos, max_cart_pos, min_bump_pos, max_bump_pos, min_bump_distance)

    state_space = indextools.JointNamedSpace(pos=pos_space)
    all_states = [
        s
        for s in state_space.elems
        if check_state(s)
    ]

    start_states = [
        s
        for s in all_states if s.pos.angle.value == 'Z'
    ]

    actions = ['LS', 'LH', 'RS', 'RH']
    action_space = indextools.DomainSpace(actions)

    obs = indextools.JointNamedSpace(
        cart=indextools.RangeSpace(min_cart_pos, max_cart_pos + 1), angle=angle_space
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
                    if (s.pos.cart.value >= min_cart_pos + 1):
                        s1.pos.cart.value -= 1

                        # small bump on the left
                        if s.pos.cart.value == s.pos.sbump.value + 1:
                            s1.pos.angle.value = 'RS'
                        # big bump on the left
                        elif s.pos.cart.value == s.pos.bbump.value + 1:
                            s1.pos.angle.value = 'RB'
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
                    if (s.pos.cart.value <= max_cart_pos - 1):
                        s1.pos.cart.value += 1

                        # small bump on the right
                        if s.pos.cart.value == s.pos.sbump.value - 1:
                            s1.pos.angle.value = 'LS'

                        # big bump on the right
                        elif s.pos.cart.value == s.pos.bbump.value - 1:
                            s1.pos.angle.value = 'LB'

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
                    if s.pos.cart.value >= min_cart_pos + 1:
                        s1.pos.cart.value -= 1

                        # small bump is here then it is moved left if angle is positive
                        if ((s.pos.cart.value == s.pos.sbump.value) and ('R' in s.pos.angle.value)):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # big bump is here then it is moved left if angle is positive
                        elif ((s.pos.cart.value == s.pos.bbump.value) and ('R' in s.pos.angle.value)):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # small bump is on the left then it is moved left if angle is zero
                        elif ((s.pos.cart.value - 1 == s.pos.sbump.value) and (s.pos.angle.value == 'Z')):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # big bump is on the left then it is moved left if angle is zero
                        elif ((s.pos.cart.value - 1 == s.pos.bbump.value) and (s.pos.angle.value == 'Z')):
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
                    if (s.pos.cart.value <= max_cart_pos - 1):
                        s1.pos.cart.value += 1

                        # big bump is here then it is moved left if finger points left
                        if ((s.pos.cart.value == s.pos.bbump.value) and ('L' in s.pos.angle.value)):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # small bump is here then it is moved left if angle is positive
                        elif ((s.pos.cart.value == s.pos.sbump.value) and ('L' in s.pos.angle.value)):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # big bump is on the right then it is moved left if angle is zero
                        elif ((s.pos.cart.value + 1 == s.pos.bbump.value) and (s.pos.angle.value == 'Z')):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        # small bump is on the left then it is moved left if angle is zero
                        elif ((s.pos.cart.value + 1 == s.pos.sbump.value) and (s.pos.angle.value == 'Z')):
                            print(f'T: {afmt(a)}: {sfmt(s)} reset')

                        else:
                            print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

                    else:
                        print(f'T: {afmt(a)}: {sfmt(s)} reset')                    

    # OBSERVATIONS
    print()
    print("O: * : * : * 0.0")
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
                    if s.pos.cart.value == s.pos.bbump.value and ('L' in s.pos.angle.value):
                        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 1.0')

                    if s.pos.cart.value == s.pos.bbump.value - 1  and (s.pos.angle.value == 'Z'):
                        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 1.0')

if __name__ == '__main__':
    main()