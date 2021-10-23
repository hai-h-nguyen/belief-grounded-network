#!/usr/bin/env python
import argparse
import copy

import indextools

# arrows: ↑ ↓ ← →

grid_base_arrows = """
→→→↓→→→→→↓
↑↓←←↑↓←←←↓
↑→→↓↑→↓→↑↓
↑↓←←↑↓←↑←←
↑→→→↑→→→→↓
↑←←←←↓←←←↓
→↓→→↑↓→↓↑↓
↑↓↑←←↓↑↓↑↓
↑→→→↑↓↑→↑↓
↑←←←←←↑←←←
"""

# converts arrows to UDLR text
grid_base = (
    grid_base_arrows.translate(str.maketrans('↑↓←→', 'UDLR'))
    .strip()
    .splitlines()
)


def reflect_h(grid):
    table = str.maketrans('LR', 'RL')
    return [row[::-1].translate(table) for row in grid]


def reflect_v(grid):
    table = str.maketrans('UD', 'DU')
    return [row.translate(table) for row in grid[::-1]]


def reverse(grid):
    grid = [list(row) for row in grid]
    nr, nc = len(grid), len(grid[0])

    rgrid = [['.' for _ in range(nc)] for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            if grid[i][j] == 'U':
                rgrid[i - 1][j] = 'D'
            elif grid[i][j] == 'D':
                rgrid[i + 1][j] = 'U'
            elif grid[i][j] == 'L':
                rgrid[i][j - 1] = 'R'
            elif grid[i][j] == 'R':
                rgrid[i][j + 1] = 'L'
            else:
                raise ValueError()

    return [''.join(row) for row in rgrid]


def get_grid(s):
    grid = copy.deepcopy(grid_base)

    if s.reflect_h.value:
        grid = reflect_h(grid)

    if s.reflect_v.value:
        grid = reflect_v(grid)

    if s.reverse.value:
        grid = reverse(grid)

    return grid


def get_tile(s):
    grid = get_grid(s)
    return grid[s.pos.y.value][s.pos.x.value]


def traverse(grid):
    nr, nc = len(grid), len(grid[0])
    visited = [[False for _ in range(nc)] for _ in range(nr)]

    i, j = 0, 0
    for _ in range(nr * nc):
        visited[i][j] = True
        if grid[i][j] == 'U':
            i -= 1
        elif grid[i][j] == 'D':
            i += 1
        elif grid[i][j] == 'L':
            j -= 1
        elif grid[i][j] == 'R':
            j += 1
        else:
            raise ValueError()

    return (i, j) == (0, 0) and all(all(row) for row in grid)


assert reflect_h(grid_base) != grid_base
assert reflect_v(grid_base) != grid_base
assert reverse(grid_base) != grid_base

assert reflect_h(reflect_h(grid_base)) == grid_base
assert reflect_v(reflect_v(grid_base)) == grid_base
assert reflect_v(reflect_h(grid_base)) == reflect_h(reflect_v(grid_base))
assert reverse(reverse(grid_base)) == grid_base

assert traverse(grid_base)
assert traverse(reflect_h(grid_base))
assert traverse(reflect_v(grid_base))
assert traverse(reverse(grid_base))


def pfmt(p):
    return f'{p.x}_{p.y}'


def sfmt(s):
    h = int(s.reflect_h.value)
    v = int(s.reflect_v.value)
    r = int(s.reverse.value)
    return f'h_{h}_v_{v}_r_{r}_pos_{s.pos.y}_{s.pos.x}'


def afmt(a):
    return a.value


def ofmt(o):
    return o.value


def main():
    parser = argparse.ArgumentParser(description='ArrowMaze')
    parser.add_argument('--gamma', type=float, default=0.99)
    config = parser.parse_args()

    pos_space = indextools.JointNamedSpace(
        x=indextools.RangeSpace(10), y=indextools.RangeSpace(10)
    )

    state_space = indextools.JointNamedSpace(
        reflect_h=indextools.BoolSpace(),
        reflect_v=indextools.BoolSpace(),
        reverse=indextools.BoolSpace(),
        pos=pos_space,
    )

    actions = 'up', 'down', 'left', 'right'
    action_space = indextools.DomainSpace(actions)

    observations = 'up', 'down', 'left', 'right'
    obs_space = indextools.DomainSpace(observations)

    print(
        """# ArrowTrail Environment;

# The agent navigates a 10x10 grid-world.  Each tile is associated with an
# arrow indicating one of the four cardinal directions;  the arrows form a path
# which covers all the tiles in a single loop, and the task is to follow the
# trail of arrows.  The agent does not observe its own position, only the
# direction indicated by the current tile.

# This environment was designed to have an easy control task and a difficult
# prediction task.

# State-space (800) : position of the agent (10x10 grid) times 8 possible paths,
# obtained from a base path through horizontal reflection, vertical reflection,
# and/or path reversal.

# Action-space (4) : directional movements {`up`, `down`, `left`, `right`}.

# Observation-space (4) : direction of the tile arrow {`up`, `down`, `left`,
# `right`}."""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {" ".join(sfmt(s) for s in state_space.elems)}')

    print(f'actions: {" ".join(afmt(a) for a in action_space.elems)}')

    print(f'observations: {" ".join(ofmt(o) for o in obs_space.elems)}')

    # # START
    # print()
    # print(f'start include: uniform')

    # TRANSITIONS
    print()
    for s in state_space.elems:
        for a in action_space.elems:
            s1 = copy.copy(s)

            if a.value == 'up':
                s1.pos.y.value = max(s1.pos.y.value - 1, 0)
            elif a.value == 'down':
                s1.pos.y.value = min(s1.pos.y.value + 1, 9)
            elif a.value == 'right':
                s1.pos.x.value = min(s1.pos.x.value + 1, 9)
            elif a.value == 'left':
                s1.pos.x.value = max(s1.pos.x.value - 1, 0)

            print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

    # OBSERVATIONS
    translation = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}

    print()
    for s1 in state_space.elems:
        tile = get_tile(s1)
        direction = translation[tile]
        o = obs_space.elem(value=direction)
        print(f'O: *: {sfmt(s1)}: {ofmt(o)} 1.0')

    # REWARDS
    print()
    print('R: *: *: *: * -10.0')
    for s in state_space.elems:
        tile = get_tile(s)
        direction = translation[tile]
        a = action_space.elem(value=direction)
        print(f'R: {afmt(a)}: {sfmt(s)}: *: * 0.0')


if __name__ == '__main__':
    main()
