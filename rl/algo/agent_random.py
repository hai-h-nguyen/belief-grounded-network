from collections import deque
import numpy as np
from rl.misc.utils import env_config
import torch

class RandomAgent:
    def __init__(self, envs, args):
        self.args = args
        self.envs = envs

        self.config = env_config(args.env_name)

        self.episode_rewards = deque(maxlen=100)

        self.envs.reset()

    def rollout(self):
        for step in range(self.args.num_steps):
            with torch.no_grad():

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