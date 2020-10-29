import numpy as np
import random

from rl.nn.envs import make_vec_envs
from rl.algo import SimulateAgent
from rl.algo import RandomAgent

class SimulateEx:
    def __init__(self, args, random_agent=False):
        self.args = args
        self.random_agent = random_agent

    def _simulate(self, n_update, num_updates):
        print("Make env simulate")
        self.sim_envs = make_vec_envs(self.args, allow_early_resets=True, seed_change=random.randint(0, 1000))

        if self.random_agent:
            self.sim_agent = RandomAgent(self.sim_envs, self.args)
        else:
            self.sim_agent = SimulateAgent(self.sim_envs, self.args)

        self.sim_agent.simulate()

    def _print(self):
        episode_rewards = self.sim_agent.get_statistic()
        if len(episode_rewards) > 1:
            print("Last {} sim episodes: mean/median reward {:.2f}/{:.2f}, min/max {:.1f}/{:.1f} std {:.2f}\n"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.std(episode_rewards)))

    def run(self, num_updates):
        for n_update in range(num_updates):
            if (self.args.eval_interval is not None and n_update % self.args.eval_interval == 0 and n_update > 0):
                self._simulate(n_update, num_updates)
                self._print()