#!/usr/bin/env python
import argparse
import itertools as itt
from copy import copy

import indextools

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

    def pstr(p):
        return f'{p.x}_{p.y}'

    def sstr(s):
        return f'{pstr(s.value.agent)}_{pstr(s.value.item)}'

    def ostr(o):
        return pstr(o)

    actions = 'query', 'left', 'right', 'up', 'down', 'buy'
    action_space = indextools.DomainSpace(actions)

    # TODO different observation space for queries and for positions..?
    # NO!  with this version, the observation makes sense in the context of
    # the action!
    obs_space = pos_space

    print(
        """# Shopping Environment;

# The agent is in a store and needs to remember which item to purchase
# (preselected at the beginning of the environment).  A reactive policy with
# insufficient memory will need to periodically query which item needs to be
# purchased.

# State-space (n ** 4) : position of the agent in the store (n ** 2 grid), and
# position of the target item in store (n ** 2 grid).

# Action-space (6) : movements in 4 directions {`left`, `right`, `up`,
# `down`}, call home to observe target item {`query`}, and purchase item in
# current cell {`buy`}.

# Observation-space (n ** 2) : if `query` action is selected, position of item;
# otherwise position of agent."""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')
    print(f'states: {state_space.nelems}')
    # print(f'states: {" ".join(sstr(s) for s in state_space.elems)}')
    print(f'actions: {" ".join(action_space.values)}')
    print(f'observations: {obs_space.nelems}')
    # print(f'observations: {" ".join(ostr(o) for o in obs_space.elems)}')

    start_states = [
        s
        for s in state_space.elems
        if s.agent.x.value == 0 and s.agent.y.value == 0
    ]
    pstart_states = 1 / len(start_states)

    # START
    print()
    print(f'start include: {" ".join(str(s.idx) for s in start_states)}')
    # print(f'start include: {" ".join(sstr(s) for s in start_states)}')
    # print(f'start include: uniform')

    # TRANSITIONS
    print()
    for a in action_space.elems:
        if a.value == 'query':
            print(f'T: {a.value} identity')
        elif a.value == 'buy':
            print(f'T: {a.value} identity')
            for s in state_space.elems:
                if s.agent == s.item:
                    print(f'T: {a.value}: {s.idx} reset')

                    # if config.episodic:
                    #     # print(f'T: {a.value}: {s.idx} {s.idx} 0.0')
                    #     print(f'T: {a.value}: {s.idx} reset')
                    #     # print(f'T: {a.value}: {sstr(s)} reset')
                    # else:
                    #     # not really uniform!.. keep same agent position!
                    #     # print(f'T: {a.value}: {s.idx} uniform')

                    #     pmatrix = [1.0 / pos_space.nelems
                    #                if s1.agent == s.agent else 0.0
                    #                for s1 in state_space.elems]
                    #     pmatrix = ' '.join(map(str, pmatrix))
                    #     # print(f'T: {a.value}: {s.idx} {s.idx} 0.0')
                    #     print(f'T: {a.value}: {s.idx} {pmatrix}')
                    #     # print(f'T: {a.value}: {sstr(s)} {pmatrix}')
        else:
            for s in state_space.elems:
                s1 = copy(s)
                if a.value == 'left' and s1.agent.x.value > 0:
                    s1.agent.x.value -= 1
                elif a.value == 'right' and s1.agent.x.value < config.n - 1:
                    s1.agent.x.value += 1
                elif a.value == 'up' and s1.agent.y.value > 0:
                    s1.agent.y.value -= 1
                elif a.value == 'down' and s1.agent.y.value < config.n - 1:
                    s1.agent.y.value += 1

                # pmatrix = [0.] * state_space.nelems
                # pmatrix[s1.idx] = 1.0
                # pmatrix = ' '.join(map(str, pmatrix))
                # print(f'T: {a.value}: {s.idx} {pmatrix}')

                print(f'T: {a.value}: {s.idx}: {s1.idx} 1.0')
                # print(f'T: {a.value}: {sstr(s)}: {sstr(s1)} 1.0')

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
        if a.value == 'query' and s1.item == o:
            print(f'O: {a.value}: {s1.idx}: {o.idx} 1.0')
            # print(f'O: {a.value}: {sstr(s1)}: {ostr(o)} 1.0')
        if a.value != 'query' and s1.agent == o:
            print(f'O: {a.value}: {s1.idx}: {o.idx} 1.0')
            # print(f'O: {a.value}: {sstr(s1)}: {ostr(o)} 1.0')

    # REWARDS
    print()
    # for a in action_space.elems:
    #     for s in state_space.elems:
    #         if a == 'query':
    #             print(f'R: {a.value}: *: *: * -2.0')
    #             break
    #         elif a == 'buy':
    #             if s.agent == s.item:
    #                 print(f'R: {a.value}: {s.idx}: *: * 10.0')
    #             else:
    #                 print(f'R: {a.value}: {s.idx}: *: * -5.0')
    #         else:
    #             print(f'R: {a.value}: *: *: * -1.0')
    #             break

    for a in action_space.elems:
        if a.value == 'query':
            print(f'R: {a.value}: *: *: * -2.0')
        elif a.value == 'buy':
            print(f'R: {a.value}: *: *: * -5.0')
            for s in state_space.elems:
                if s.agent == s.item:
                    print(f'R: {a.value}: {s.idx}: *: * 10.0')
                    # print(f'R: {a.value}: {sstr(s)}: *: * 10.0')
        else:
            print(f'R: {a.value}: *: *: * -1.0')
