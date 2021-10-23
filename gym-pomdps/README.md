# gym-pomdps

This repository contains gym environments for flat/discrete POMDPs loaded from
the pomdp file format.

## Installation

This package is dependent on the
[rl_parsers](https://github.com/abaisero/rl_parsers) package.  Install
`rl_parsers` before proceeding.

## Contents

This repository provides the `POMDP` environment and the `BatchPOMDP` wrappers.

### POMDP Environment

The POMDP environment receives a path to the pomdp file, and a boolean flag
indicating whether the POMDP should be considered episodic or continuing (more
on this later).

All the POMDPs in the `pomdps/` folder are registered under gym:

 * An episodic variant under ID `POMDP-{name}-episodic-v{version}`; and
 * A continuing variant under ID `POMDP-{name}-continuing-v{version}`.

#### Episodic and Continuing Environments

The `reset` keyword in pomdp files (see
[rl_parsers](https://github.com/abaisero/rl_parsers) for details) denotes the
end of a sequential experience.  This library uses that in two ways:

 * In the episodic variant, the terminal condition has been reached and the
   sequential experience has concluded.
 * In the continuing variant, the state is resampled from the initial
   distribution, and the sequential experience indefinitely.

### BatchPOMDP Wrapper

The BatchPOMDP wrapper runs multiple independent (but synchronized) experiences
at the same time, and is more efficient than running the experiences
sequentially.  The wrapper receives a POMDP environment and the number of
experiences to run concurrently.  States, actions, observations, rewards and
dones are vectorized (with np.array).

NOTE:  This wrapper currently only supports continuing POMDPs.
