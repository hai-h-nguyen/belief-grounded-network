import numpy as np
import random

from rl.nn.envs import make_vec_envs
from rl.algo import Agent
from rl.algo import TestBCAgent

class CloneEx:
    def __init__(self, args):
        self.args = args

        assert(self.args.policy_file is not None)
        assert(self.args.transitions_file is not None)
        assert(self.args.algo in ['bc', 'bc-embed'])

        # Make sure parameters are matched
        if 'embed' in self.args.policy_file:
            assert 'embed' in self.args.algo

        print("Make env cloning")
        self.clone_envs = make_vec_envs(self.args, allow_early_resets=False)
        self.clone_agent = Agent(self.clone_envs, self.args, self.args.device)
        self.clone_agent.relabel_actions()

    def _clone(self):
        return self.clone_agent.clone()

    def _evaluate(self):
        self.args.num_processes = 4
        self.test_envs = make_vec_envs(self.args, allow_early_resets=True, seed_change=random.randint(0, 1000))
        self.test_agent = TestBCAgent(self.test_envs, self.args, self.args.device, self.clone_agent.actor_critic.rnn_state_size)    
        self.test_agent.evaluate(self.clone_agent.cloner)    

    def _print(self, update_str):
        episode_rewards = self.test_agent.get_statistic()
        if len(episode_rewards) > 1:
            print(update_str)
            print("Last {} BC episodes: mean/median reward {:.2f}/{:.2f}, min/max {:.1f}/{:.1f} std {:.2f}\n"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.std(episode_rewards)))

    def run(self):
        for n_update in range(self.args.bc_num_epochs):
            self._clone()

            if n_update > 0:
                status_str = "Behavior Cloning Updates {}".format(n_update)
                self._evaluate()
                self._print(status_str)        