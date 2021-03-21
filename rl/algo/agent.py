from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import sys
import pickle
import bz2
import dill
from copy import deepcopy

from .actor_critic_methods import ABCB, AHCB, AHCH, AHCS, ASCS
from .bc import BehaviorClone

from rl.nn import A2CAsym, A2C, A2CAsymDual
from rl.nn import BC
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

        # Using belief loss, saved file format is different
        if self.args.algo == 'ah-ch' and self.args.belief_loss_coef > 0.0:
            self.model_path = os.path.join(log_dir, self.args.algo + '.b' + '.' + str(self.args.seed) + '.mdl')
            self.transitions_path = os.path.join(log_dir, self.args.env_name + '.' +
                                                                       self.args.algo + '.b' + '.' + 
                                                                       str(self.args.seed) + '.exp')
        else:
            self.model_path = os.path.join(log_dir, self.args.algo + '.' + str(self.args.seed) + '.mdl')
            self.transitions_path = os.path.join(log_dir, self.args.env_name + '.' +
                                                                       self.args.algo + '.' + 
                                                                       str(self.args.seed) + '.exp')                                                                            
        self.setup_agent()

    def setup_agent(self):
        algo = self.args.algo

        n_reactive = self.args.n_reactive
        if algo != 'af-cf':
            assert n_reactive == 1, 'only reactive policy should use --n-reactive'        

        if algo == 'ah-cs':  # Recurrent actor, true-state critic
            self.actor_critic = A2CAsym(self.config, actor_recurrent=True, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCS(self.actor_critic, self.args, self.config)

        if algo == 'ab-cb':  # Belief actor, belief critic
            dims = (self.config['belief_dim'], self.config['belief_dim'], self.config['action_size'], self.config['n_known_states'])
            self.actor_critic = A2CAsymDual(dims, critic_recurrent=False, actor_recurrent=False, use_embedding=False).to(self.device)
            self.agent = ABCB(self.actor_critic, self.args, self.config)

        if algo == 'as-cs':  # true-state actor, true-state critic
            dims = (self.config['state_dim'], self.config['state_dim'], self.config['action_size'], self.config['n_known_states'])
            self.actor_critic = A2CAsymDual(dims, critic_recurrent=False, actor_recurrent=False, use_embedding=False, embed_actions=False).to(self.device)
            self.agent = ASCS(self.actor_critic, self.args, self.config)

        if algo == 'bc':  # Behavioral cloning
            dims = (self.config['belief_dim'], self.config['belief_dim'], self.config['action_size'], self.config['n_known_states'])
            self.cloner = BC(self.config, use_embedding=self.should_embed).to(self.device)
            self.clone_agent = BehaviorClone(self.cloner, self.args, self.config)
            self.actor_critic = A2CAsymDual(dims, actor_recurrent=False, critic_recurrent=False, use_embedding=self.should_embed, embed_actions=False).to(self.device)

        if algo == 'ah-cb':  # Recurrent actor, belief critic
            if not self.should_embed:
                dims = (self.config['obs_dim'], self.config['belief_dim'], self.config['action_size'], self.config['n_known_states'])
            else:            
                dims = (self.config['belief_dim'], self.config['obs_size'], self.config['action_size'], self.config['n_known_states'])
            self.actor_critic = A2CAsymDual(dims, actor_recurrent=True, critic_recurrent=False, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCB(self.actor_critic, self.args, self.config)


        if algo == 'ah-ch':  # Recurrent actor, recurrent critic
            dims = (self.config['obs_size' if self.should_embed else 'obs_dim'],
                    self.config['action_size'], self.config['belief_dim'], self.config['n_known_states'])            
            self.actor_critic = A2C(dims, n_reactive, recurrent=True, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCH(self.actor_critic, self.args, self.config)

            if self.args.policy_file is not None:
                self.actor_critic = torch.load(self.args.policy_file)
                print("Load pre-trained policy")


        if algo == 'af-cf':  # Reactive actor, reactive critic (i.e. fixed-length memory)
            dims = (self.config['obs_size' if self.should_embed else 'obs_dim'],
                    self.config['action_size'],
                    self.config['belief_dim'],
                    self.config['n_known_states'])
            self.actor_critic = A2C(dims, n_reactive, recurrent=False, use_embedding=self.should_embed).to(self.device)
            self.agent = AHCH(self.actor_critic, self.args, self.config)

        if not hasattr(self, 'actor_critic'):
            raise ValueError('{} is not a valid algorithm name'.format(algo))

        self.rollouts = RolloutStorage(self.args, self.config, self.actor_critic.rnn_state_size, self.should_embed)
        obs = self.envs.reset()
        state = self.envs.get_state()
        belief = self.envs.get_belief()

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.state[0].copy_(state)
        self.rollouts.belief[0].copy_(belief)

        self.rollouts.to(self.device)   

    def rollout(self):
        for step in range(self.args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, actor_hidden, critic_hidden = self.actor_critic.act(self.rollouts, step, self.args)

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

        # Only supports these types of experts
        if self.args.algo in ['ab-cb', 'as-cs'] and self.args.save_transitions:
            with torch.no_grad():
                self.experience_mem.append([self.rollouts.obs.clone().numpy(), 
                                            self.rollouts.masks.clone().numpy(), 
                                            self.rollouts.belief.clone().numpy(), 
                                            self.rollouts.actions.clone().numpy(), 
                                            self.rollouts.state.clone().numpy(),
                                            ])

            if n_update == num_updates - 1 or (n_update % 1000 == 0 and n_update > 0):
                print("Saving ...")
                sfile = bz2.BZ2File(self.transitions_path, "wb")
                dill.dump(self.experience_mem, sfile)                
            
        self.rollouts.after_update()

    def clone(self):
        for i_batch, sample_batched in enumerate(self.data_loader):
            loss = self.clone_agent.update(sample_batched)
            if i_batch % 20 == 0:
                print("BC loss: {:.3f}".format(loss))           

    def relabel_actions(self):

        # Load expert
        self.actor_critic = torch.load(self.args.policy_file)
        self.actor_critic.eval()

        # Load transitions
        pickle_file = bz2.BZ2File(self.args.transitions_file, 'rb')
        self.experience_mem = pickle.load(pickle_file)

        print("Relabel actions {} transitions ...".format(len(self.experience_mem)))
        for i in tqdm(range(len(self.experience_mem))):
            for step in range(self.args.num_steps):
                with torch.no_grad():

                    # Ab-Cb expert, input_idx=2 is beliefs
                    actions = self.actor_critic.act_simple(self.experience_mem[i], step, self.args, 
                                                            input_idx=2, 
                                                            deterministic=True)

                    # Override old actions w/ expert actions                                                   
                    self.experience_mem[i][3][step] = actions.clone()

        print("Done!")

        self.data_loader = DataLoader(self.experience_mem, batch_size=self.args.bc_batch_size, shuffle=True, drop_last=True)

    def load_expert_data(self):
        # Load transitions
        pickle_file = bz2.BZ2File(self.args.transitions_file, 'rb')
        self.experience_mem = pickle.load(pickle_file)
        self.data_loader = DataLoader(self.experience_mem, batch_size=self.args.bc_batch_size, shuffle=True, drop_last=True)        

    def get_statistic(self):
        return self.episode_rewards, self.value_loss, self.action_loss, self.dist_entropy

    def save(self):
        torch.save(self.actor_critic, self.model_path)