#!/usr/bin/env python
import argparse
import itertools as itt
from copy import copy

import indextools


def pfmt(p):
    return f'{p.x}_{p.y}'


def sfmt(s):
    return f'agent_{pfmt(s.agent)}_item_{pfmt(s.item)}'


def afmt(a):
    return a.value


def ofmt(o):
    return f'{o.postype}_{pfmt(o.pos)}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shopping')
    parser.add_argument('n', type=int, default=None)
    # parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    config = parser.parse_args()

    # TODO change size to width and height
    assert config.n > 1
    assert 0 < config.gamma <= 1

    pos_space = indextools.JointNamedSpace(
        x=indextools.RangeSpace(config.n), y=indextools.RangeSpace(config.n)
    )

    state_space = indextools.JointNamedSpace(agent=pos_space, item=pos_space)

    actions = 'query', 'left', 'right', 'up', 'down', 'buy'
    action_space = indextools.DomainSpace(actions)

    postypes = 'agent', 'item'
    postype_space = indextools.DomainSpace(postypes)
    obs_space = indextools.JointNamedSpace(postype=postype_space, pos=pos_space)

    # print('states')
    # for s in state_space.elems:
    #     print(sfmt(s))

    # print('actions')
    # for a in action_space.elems:
    #     print(afmt(a))

    # print('observations')
    # for o in obs_space.elems:
    #     print(ofmt(o))

    # import sys
    # sys.exit(0)

    print(
        """# Shopping Environment;

# The agent navigates a gridworld store with the goal of purchasing an item at
# an unknown position.  Observations regarding the object position can be
# obtained and should ideally be memorized during navigation---a reactive
# policy with insufficient memory capabilities will need to periodically query
# the target item position.

# State-space (n ** 4) : position of the agent in the store (n ** 2 grid), and
# position of the target item in store (n ** 2 grid).

# Action-space (6) : directional movements {`left`, `right`, `up`, `down`},
# query info on target item position {`query`}, and purchase item in current
# cell {`buy`}.

# Observation-space (n ** 4) : position of the item if `query` action is
# selected, otherwise position of the agent."""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    # print(f'states: {state_space.nelems}')
    print(f'states: {" ".join(sfmt(s) for s in state_space.elems)}')

    # print(f'actions: {action_space.nelems}')
    # print(f'actions: {" ".join(action_space.values)}')
    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')

    # print(f'observations: {obs_space.nelems}')
    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    start_states = [
        s
        for s in state_space.elems
        if s.agent.x.value == 0 and s.agent.y.value == 0
    ]
    # pstart_states = 1 / len(start_states)

    # START
    print()
    # print(f'start include: {" ".join(str(s.idx) for s in start_states)}')
    print(f'start include: {" ".join(sfmt(s) for s in start_states)}')
    # print(f'start include: uniform')

    # TRANSITIONS
    print()
    for a in action_space.elems:
        if a.value == 'query':
            print(f'T: {afmt(a)} identity')
        elif a.value == 'buy':
            print(f'T: {afmt(a)} identity')
            for s in state_space.elems:
                if s.agent == s.item:
                    print(f'T: {afmt(a)}: {sfmt(s)} reset')
        else:
            for s in state_space.elems:
                s1 = copy(s)
                if a.value == 'left':
                    s1.agent.x.value = max(s1.agent.x.value - 1, 0)
                elif a.value == 'right':
                    s1.agent.x.value = min(s1.agent.x.value + 1, config.n - 1)
                elif a.value == 'up':
                    s1.agent.y.value = min(s1.agent.y.value + 1, config.n - 1)
                elif a.value == 'down':
                    s1.agent.y.value = max(s1.agent.y.value - 1, 0)

                print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

    # OBSERVATIONS
    print()
    # for a, s1, o in itt.product(action_space.elems, state_space.elems,
    #                             obs_space.elems):
    #     if a == 'query':
    #         if s1.item == o:
    #             print(f'O: {a.value}: {s1.idx}: {o.idx} 1.0')
    #     elif a == 'buy':
    #         if s1.item
    #     else:

    #     if a != 'query' and s1.agent == o:
    #         print(f'O: {a.value}: {s1.idx}: {o.idx} 1.0')
    for a, s1, o in itt.product(
        action_space.elems, state_space.elems, obs_space.elems
    ):
        if (
            a.value == 'query'
            and o.postype.value == 'item'
            and s1.item == o.pos
        ):
            print(f'O: {afmt(a)}: {sfmt(s1)}: {ofmt(o)} 1.0')
        if (
            a.value != 'query'
            and o.postype.value == 'agent'
            and s1.agent == o.pos
        ):
            print(f'O: {afmt(a)}: {sfmt(s1)}: {ofmt(o)} 1.0')

    # REWARDS
    print()
    for a in action_space.elems:
        if a.value == 'query':
            print(f'R: {afmt(a)}: *: *: * -2.0')
        elif a.value == 'buy':
            print(f'R: {afmt(a)}: *: *: * -50.0')
            for s in state_space.elems:
                if s.agent == s.item:
                    print(f'R: {afmt(a)}: {sfmt(s)}: *: * 100.0')
        else:
            print(f'R: {afmt(a)}: *: *: * -1.0')
