# Belief-Grounded Networks for Accelerated Robot Learning under Partial Observability

This is the repo stored the code for our paper [Belief-Grounded Network for Accelerated Robot Learning under Partial Observability](https://arxiv.org/abs/2010.09170) accepted at CoRL 2020. This is a joint repo contributed with [Brett](https://github.com/brett-daley) and [Song](https://github.com/xinchaosong). Project [website](https://sites.google.com/view/bgn-pomdp). If you use this repository in published work, please cite the paper:

```

@InProceedings{pmlr-v155-nguyen21a,
  title = 	 {Belief-Grounded Networks for Accelerated Robot Learning under Partial Observability},
  author =       {Nguyen, Hai and Daley, Brett and Song, Xinchao and Amato, Christopher and Platt, Robert},
  booktitle = 	 {Proceedings of the 2020 Conference on Robot Learning},
  pages = 	 {1640--1653},
  year = 	 {2021},
  editor = 	 {Kober, Jens and Ramos, Fabio and Tomlin, Claire},
  volume = 	 {155},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {16--18 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v155/nguyen21a/nguyen21a.pdf},
  url = 	 {https://proceedings.mlr.press/v155/nguyen21a.html},
  abstract = 	 {Many important robotics problems are partially observable where a single visual or force-feedback measurement is insufficient to reconstruct the state. Standard approaches involve learning a policy over beliefs or observation-action histories.    However, both of these have drawbacks; it is expensive to track the belief online, and it is hard to learn policies directly over histories. We propose a method for policy learning under partial observability called the Belief-Grounded Network (BGN) in which an auxiliary belief-reconstruction loss incentivizes a neural network to concisely summarize its input history. Since the resulting policy is a function of the history rather than the belief, it can be executed easily at runtime. We compare BGN against several baselines on classic benchmark tasks as well as three novel robotic force-feedback tasks. BGN outperforms all other tested methods and its learned policies work well when transferred onto a physical robot.}
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

* Algorithm names: `ab-cb, ah-cb, ah-ch, ah-cs`
* Domain names: `PomdpHallway-v0, PomdpHallway-v2, PomdpRs44-v0, PomdpRS55-v0, MomdpBumps-v0, MomdpPlates-v0, MomdpTopPlate-v0`
* Running modes: train, simulate (replay a policy)
* Command (tee is to save the output to a file for plotting later): 
  * Train: 
  ```
  python3 -u main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode train --seed 0 | tee log.txt
  ```
  * Simulate a saved policy: 
  ```
  python3 main.py --algo algo-name --num-env-steps num-steps --seed 0 --env-name name --running-mode simulate --policy-file file --eval-interval 100
  ```
  * Runing BGN w/ an ah-ch agent: 
  ```
  python3 -u main.py --algo ah-ch --num-env-steps num-steps --seed 0 --env-name name --running-mode train --belief-loss-coef 1.0 | tee log.txt
  ```
* For all training commands, the policy will be autonomously saved at `scripts/logs/env-name/algo-name.#seed.mdl`

---

## Visualize

* Plot using the script in folder `plot` which takes a text file as the input with the option to plot training/validation results, smooth window:
  * Plot a single folder: sub-folders must have names such as `ahcb, abcb, ahcs, ahch, bgn`, each contain the runs for different seeds: 
  ```
  python3 plot_folder.py --folder hallway --window 10 --mode training/testing
  ```
  * Plot multiple folders: 
  ```
  python3 plot_folders.py --folder hallway hallway2 rs44 rs55 --window 10 10 10 10 --mode testing testing training training
  ```

---

## License

This code is released under the [MIT License](https://github.com/hai-h-nguyen/belief-grounded-network/blob/master/LICENSE).

---

## Acknowledgments

This codebase evolved from the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) but heavily modified.
