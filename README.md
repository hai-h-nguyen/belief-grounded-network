# Belief-Grounded Networks for AcceleratedRobot Learning under Partial Observability

This is the repo stored the code for our paper [Belief-Grounded Network for Accelerated Robot Learning under Partial Observability](https://arxiv.org/abs/2010.09170) accepted at CoRL 2020. If you use this repository in published work, please cite the paper:

```
@article{nguyen2020belief,
  title={Belief-Grounded Networks for AcceleratedRobot Learning under Partial Observability},
  author={Nguyen, Hai and Daley, Brett and Song, Xinchao and Amato, Chistopher and Platt, Robert},
  journal={arXiv preprint arXiv:2010.09170},
  year={2020}
}
```
---
## Contents

[Setup](#setup)

[Train](#train)

[Visualize](#visualize)

[License, Acknowledgments](#license)

---

## Setup

- Install gym-pomdps from `https://github.com/abaisero/gym-pomdps` by `pip install -e .`
- Install dependency: `pip install -r requirements.txt`
- Install MuJoCo
- After that
  - Copy `.pomdp` domain files in folder `domain` to `gym_pomdps/pomdps`
  - Copy the domain folder in folder `domain` to `gym/envs/`
  - Register new domains with `gym` by adding the content in `modifications/__init__.py` to `gym/envs/__init__.py`
  - Modify several `baselines` files according to files in the folder `modifications`
    * `bench/monitor.py` - adding discounted reward calculation
    * `common/vec_env/dummy_vec_env.py` - adding get states and get belief functions
    * `common/vec_env/shmem_vec_env.py` - adding get states and get belief functions
  - Modify line 96 in `gym-pomdps\gym_pomdps\pomdp.py` from `state_next = -1` to `state_next = self.state_space.n`

---

## Train

* algo-name: `ab-cb, ah-cb, ah-ch, ah-cs, bc` for any domain
* domain-name: `PomdpHallway-v0, PomdpHallway-v2, PomdpRs44-v0, PomdpRS55-v0, MomdpBumps-v0, MomdpPlates-v0, MomdpTopPlate-v0`
* running-mode: train, simulate (a saved policy file or a random agent), or clone an ab-cb epxert (need to have a policy file and an experience file saved before)
* Command (tee is to save the output to a file for plotting later): 
  * Train: 
  ```
  python3 -u main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode train | tee log.txt
  ```
  * Simulate a saved policy: 
  ```
  python3 main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode simulate --policy-file file --eval-interval 100
  ```
  * Runing BGN w/ an ah-ch agent: 
  ```
  python3 -u main.py --algo ah-ch --num-env-steps num-steps --seed 0 --env-name name --running-mode train --belief-loss-coef 1.0 | tee log.txt
  ```
  * For all training commands, the policy will be autonomously saved at `scripts/logs/env-name/algo-name.seed.mdl`

---

## Visualize

* Plot using the script in folder `plot` which takes `file.txt` as the input with the option to plot training and/or validating results:
  * Plot a folder for a domain: sub-folders must have names such as `ahcb, abcb, ahcs, ahch, gbn`: `python3 plot_folder.py --folder hallway --window 10 --mode training/testing`
  * Plot multiple folders corresponding to four domains: 
  ```
  python3 plot_folders.py --folder hallway hallway2 rs44 rs55 --window 10 10 10 10 --mode testing testing training training
  ```

---

## License

This code is released under the MIT License.

---

## Acknowledgments

This codebase evolved from the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) but heavily modified.