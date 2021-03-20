from collections import deque
import numpy as np
from rl.misc.utils import env_config
import torch
from scipy.stats import entropy

# A heuristic agent for HeavenHell
class HeuristicAgent:
    def __init__(self):
        self.episode_rewards = deque(maxlen=100)

    def _compute_action(self, state, belief):

        current_entropy = entropy(belief)

        # Go directly to heaven
        if current_entropy == 0:
            if state in [7, 0, 1, 17, 10, 11]:
                return 0

            if state in [2, 3, 5, 8, 9, 18, 19]:
                return 3

            if state in [12, 13, 15]:
                return 2     

            return 0

        # Go to the priest
        else:
            if state in [3, 13, 17, 18, 7, 8]:
                return 2

            if state in [5, 15]:
                return 3

            if state in [2, 1, 0, 12, 11, 10]:
                return 1

            return 0

    def rollout(self):
        for step in range(self.args.num_steps):
            with torch.no_grad():

                action = self.actor_critic.act(self.rollouts, step, self.args)

                # Select random actions
                action = np.random.randint(self.config['action_size'], size=(self.args.num_processes, 1))
                action = torch.LongTensor(action)

            # Observe reward and next obs
            _, _, done, infos = self.envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])

    def simulate(self):
        while len(self.episode_rewards) < 100:
            self.rollout()

        self.envs.close()

    def get_statistic(self):
        return self.episode_rewards