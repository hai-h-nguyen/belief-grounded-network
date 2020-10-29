from collections import deque
import torch
import os
import numpy as np

from rl.nn.envs import should_embed
from rl.storage import RolloutStorage
from rl.misc.utils import env_config

class TestBCAgent:
    def __init__(self, envs, args, device, rnn_state_size):
        self.device = device
        self.args = args
        self.envs = envs

        self.episode_rewards = deque(maxlen=100)

        self.config = env_config(args.env_name)

        self.is_embed = should_embed(args.env_name)

        self.rollouts = RolloutStorage(self.args, self.config, rnn_state_size, self.is_embed)
        obs = self.envs.reset()

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)              

    def rollout(self, bc_agent):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                actions, actor_hidden = bc_agent.act(self.rollouts, step, deterministic=True)

            # Observe reward and next obs
            obs, _, done, infos = self.envs.step(actions)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            self.rollouts.insert_minimal(obs, actor_hidden, actions, masks)            

    def evaluate(self, agent):
        while len(self.episode_rewards) < 100:
            self.rollout(agent)
            self.rollouts.after_update()

        self.envs.close()

    def get_statistic(self):
        return self.episode_rewards