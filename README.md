# Exploration

## Contents

[Setup](#setup)

[Train](#train)

---

## Setup

- Install gym-pomdps from `https://github.com/abaisero/gym-pomdps` by `pip install -e .`
- Install dependency: `pip install -r requirements.txt`
- Install MuJoCo
- After that
  - Copy `.pomdp` domain files in folder `domains/pomdp_files` to `gym_pomdps/pomdps`
  - Copy the domains' folders in `domains/pomdp_files` to `gym/envs/`
  - Register new domains with `gym` by adding the content in `modifications/__init__.py` to `gym/envs/__init__.py`
  - Modify several `baselines` files as in the folder `modifications`
    * `baselines/bench/monitor.py` - adding discounted reward calculation
    * `baselines/common/vec_env/dummy_vec_env.py` - adding get states and get belief functions
    * `baselines/common/vec_env/shmem_vec_env.py` - adding get states and get belief functions
  - Modify line 96 in `gym-pomdps/gym_pomdps/pomdp.py` from `state_next = -1` to `state_next = self.state_space.n`

---

## Train

* Algorithm names: `ah-ch, ah-cs`
* Domain names: `PomdpHallway-v0, PomdpHallway2-v0, PomdpHeavenHell-v0`
* Running modes: train, simulate (replay a policy)
* Command (tee is to save the output to a file for plotting later): 
  * Train: 
  ```
  python3 -u main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode train --seed 0
  ```
  * Simulate a saved policy: 
  ```
  python3 main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode simulate --policy-file file --eval-interval 100
  ```
* For all training commands, the policy will be autonomously saved at `scripts/logs/env-name/algo-name.#seed.mdl`

---

## Acknowledgments

This codebase evolved from the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) but heavily modified.