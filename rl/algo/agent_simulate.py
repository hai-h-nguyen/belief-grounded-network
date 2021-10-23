from collections import deque
import torch
from torch import nn as nn
import os
import numpy as np

from rl.nn.envs import should_embed
from rl.storage import RolloutStorage
from rl.misc.utils import env_config

class SimulateAgent:
    def __init__(self, envs, args):
        self.args = args
        self.envs = envs

        assert(self.args.policy_file is not None)

        self.episode_rewards = deque(maxlen=100)

        self.config = env_config(args.env_name)

        self.is_embed = should_embed(args.env_name)

        self.actor_critic = torch.load(self.args.policy_file)

        # We want to act in each step
        self.args.num_steps = 1

        self.rollouts = RolloutStorage(self.args, self.config, self.actor_critic.rnn_state_size, self.is_embed)

        obs = self.envs.reset()
        state = self.envs.get_state()

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.state[0].copy_(state)

        self.rollouts.to('cpu')    

    def _prepare_inputs(self):
        obs_shape = self.rollouts.obs.size()[2:]
        action_shape = self.rollouts.actions.size()[-1]
        return (self.rollouts.obs[:-1].view(-1, *obs_shape),
                self.rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                self.rollouts.masks[:-1].view(-1, 1),
                self.rollouts.actions.view(-1, action_shape))

    def rollout(self, actor_critic):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, actor_hidden, critic_hidden = actor_critic.act(self.rollouts, step, self.args, deterministic=True)
                inputs = self._prepare_inputs()

            # Observe reward and next obs
            obs, reward, done, infos = self.envs.step(action)

            state_ts = torch.empty((self.args.num_processes, self.config['state_dim']), dtype=torch.float)
            index = 0

            for info in infos:
                state_ts[index] = torch.FloatTensor(info['curr_state'])
                index += 1
                    
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            self.rollouts.insert(obs, state_ts, actor_hidden, critic_hidden, action,
                        action_log_prob, value, reward, masks, bad_masks)            

    def simulate(self):
        while len(self.episode_rewards) < 100:
            self.rollout(self.actor_critic)
            self.rollouts.after_update()

        self.envs.close()

    def get_statistic(self):
        return self.episode_rewards