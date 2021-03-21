from collections import deque
import torch
import os
import numpy as np

from rl.nn.envs import should_embed
from rl.storage import RolloutStorage
from rl.misc.utils import env_config

class TestAgent:
    def __init__(self, envs, args, rnn_state_size):
        self.device = args.device
        self.args = args
        self.envs = envs

        self.episode_rewards = deque(maxlen=100)

        self.config = env_config(args.env_name)

        self.is_embed = should_embed(args.env_name)

        self.rollouts = RolloutStorage(self.args, self.config, rnn_state_size, self.is_embed)

        obs = self.envs.reset()
        state = self.envs.get_state()
        belief = self.envs.get_belief()

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.state[0].copy_(state)
        self.rollouts.belief[0].copy_(belief)

        self.rollouts.to(self.device)              

    def rollout(self, actor_critic):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, actor_hidden, critic_hidden = actor_critic.act(self.rollouts, step, self.args, deterministic=True)

            # Observe reward and next obs
            obs, reward, done, infos = self.envs.step(action)

            state_ts = torch.empty((self.args.num_processes, self.config['state_dim']), dtype=torch.float)
            belief_ts = torch.empty((self.args.num_processes, self.config['belief_dim']), dtype=torch.float)
            index = 0

            for info in infos:
                state_ts[index] = torch.FloatTensor(info['curr_state'])
                belief_ts[index] = torch.FloatTensor(info['belief'])
                index += 1
                    
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r_e'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            self.rollouts.insert(obs, state_ts, belief_ts, actor_hidden, critic_hidden, action,
                        action_log_prob, value, reward, masks, bad_masks)            

    def evaluate(self, actor_critic):
        while len(self.episode_rewards) < 100:
            self.rollout(actor_critic)
            self.rollouts.after_update()

        self.envs.close()

    def get_statistic(self):
        return self.episode_rewards