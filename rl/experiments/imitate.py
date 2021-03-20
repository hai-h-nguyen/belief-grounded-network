import numpy as np
import random

from rl.nn.envs import make_vec_envs
from rl.algo import Agent
from rl.algo import TestBCAgent

import wandb

class ImitateEx:
    def __init__(self, args):
        self.args = args

        print("Make env imitating")
        self.imitate_envs = make_vec_envs(self.args, allow_early_resets=False)
        self.imitate_agent = Agent(self.imitate_envs, self.args, self.args.device)
        self.imitate_agent.load_expert_data()

    def _imitate(self):
        return self.imitate_agent.clone()

    def _evaluate(self):
        self.args.num_processes = 1
        self.test_envs = make_vec_envs(self.args, allow_early_resets=True, seed_change=random.randint(0, 1000))
        self.test_agent = TestBCAgent(self.test_envs, self.args, self.args.device, self.imitate_agent.actor_critic.rnn_state_size)    
        self.test_agent.evaluate(self.imitate_agent.cloner)    

    def _print(self, n_update, log_wandb):
        episode_rewards = self.test_agent.get_statistic()
        update_str = "Updates Imitate {}".format(n_update)
        if len(episode_rewards) > 1:
            print(update_str)
            print("Last {} Imitation episodes: mean/median reward {:.2f}/{:.2f}, min/max {:.1f}/{:.1f} std {:.2f}\n"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.std(episode_rewards)))

        if log_wandb:
            wandb.log({'Average Return': np.mean(episode_rewards)}, 
                        step=n_update)    

    def run(self):
        if self.args.wandb:
            wandb.init(project='BGN-Test', group=self.args.group,
               name='_'.join([self.args.algo, self.args.env_name, 'seed=' + str(self.args.seed)]))

        for n_update in range(self.args.bc_num_epochs):
            self._imitate()

            if n_update > 0:
                self._evaluate()
                self._print(n_update, self.args.wandb)        