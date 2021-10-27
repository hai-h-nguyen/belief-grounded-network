# Exploration

## Contents

[Setup](#setup)

[Train](#train)

---

## Setup

- Install gym-pomdps in the corresponding folder by `pip install -e .`
- Install bgn: `pip install -r requirements.txt`
- Install MuJoCo
- After that
  - Copy the domains' folders in `domains/pomdp_files` to `gym/envs/`
  - Install openai baselines and override the following files with the ones inside `modifications/`
    * `baselines/bench/monitor.py` - adding discounted reward calculation
    * `baselines/common/vec_env/dummy_vec_env.py` - adding get states
    * `baselines/common/vec_env/shmem_vec_env.py` - adding get states

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