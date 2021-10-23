from collections import deque
import torch
import os
import numpy as np
from copy import deepcopy

from .actor_critic_methods import AHCH, AHCS

from rl.nn import A2CAsym, A2C
from rl.nn.envs import should_embed
from rl.storage import RolloutStorage
from rl.misc.utils import env_config
from rl.misc.scheduling import LinearSchedule, ExponentialSchedule

class Agent:
    def __init__(self, envs, args, log_dir):
        self.device = args.device
        self.args = args
        self.envs = envs

        self.episode_rewards = deque(maxlen=100)

        self.config = env_config(args.env_name)

        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

        if self.args.use_linear_entropy_decay:
            self.entropy_schedule = ExponentialSchedule(args.entropy_coef, 2e-5, num_updates)

        self.should_embed = should_embed(args.env_name)

        self.experience_mem = []

        self.model_path = os.path.join(log_dir, self.args.algo + '.' + str(self.args.seed) + '.mdl')

        self.setup_agent()

    def setup_agent(self):
        algo = self.args.algo

        if algo == 'ah-cs':  # Recurrent actor, true-state critic
            self.actor_critic = A2CAsym(self.config, actor_recurrent=True, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCS(self.actor_critic, self.args, self.config)

        if algo == 'ah-ch':  # Recurrent actor, recurrent critic
            dims = (self.config['obs_size' if self.should_embed else 'obs_dim'],
                    self.config['action_size'])            
            self.actor_critic = A2C(dims, recurrent=True, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCH(self.actor_critic, self.args, self.config)

        if not hasattr(self, 'actor_critic'):
            raise ValueError('{} is not a valid algorithm name'.format(algo))

        self.rollouts = RolloutStorage(self.args, self.config, self.actor_critic.rnn_state_size, self.should_embed)
        obs = self.envs.reset()
        state = self.envs.get_state()

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.state[0].copy_(state)

        self.rollouts.to(self.device)   

    def rollout(self):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, actor_hidden, critic_hidden = self.actor_critic.act(self.rollouts, step, self.args)

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

    def compute_returns(self):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts, self.args).detach()

        self.args.gamma = self.config['discount']
        self.rollouts.compute_returns(next_value, self.args)

    def train(self, n_update, num_updates):

        if self.args.use_linear_entropy_decay:
            entropy_coef = self.entropy_schedule.value(n_update)
            self.agent.update_entropy_coef(entropy_coef)

        self.rollout()
        self.compute_returns()
        self.value_loss, self.action_loss, self.dist_entropy = self.agent.update(self.rollouts)

        self.rollouts.after_update()

    def get_statistic(self):
        return self.episode_rewards, self.value_loss, self.action_loss, self.dist_entropy

    def save(self):
        torch.save(self.actor_critic, self.model_path)