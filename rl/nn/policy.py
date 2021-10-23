import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.misc.distributions import Categorical
from rl.misc.utils import init, bind
from rl.nn.second_base import MLPBase, MLPBaseAsym

# Separating backbone between actor and critic
class A2C(nn.Module):
    def __init__(self, dims, recurrent, use_embedding, embed_size=128):
        super(A2C, self).__init__()
        self.use_embedding = use_embedding

        if use_embedding:
            self.embed_size = embed_size
            self.action_embed = nn.Embedding(dims[1], embed_size)
            self.obs_embed = nn.Embedding(dims[0], embed_size)
            mlp_input_size = 2 * embed_size
        else:
            mlp_input_size = dims[0]

        self.base = MLPBase(mlp_input_size, recurrent=recurrent)
        self.dist = Categorical(self.base.output_size, dims[1])

    @property
    def rnn_state_size(self):
        """Size of rnn_hx."""
        return self.base.rnn_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _embed_if_needed(self, obs):
        if not self.use_embedding:
            return obs

        obs = obs.view(obs.shape[0], -1)        
        actions_embedded = self.action_embed(obs[:, 1])
        obs_embedded = self.obs_embed(obs[:, 0])
        return bind(actions_embedded, obs_embedded).reshape(-1, 2 * self.embed_size)

    def act(self, rollouts, step, args, deterministic=False):
        obs_inputs = rollouts.obs[step]
        rnn_hxs = rollouts.actor_rnn_states[step]
        masks = rollouts.masks[step]

        inputs = self._embed_if_needed(obs_inputs)
        value, actor_features, _, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs, None

    def get_value(self, rollouts, args):
        obs_inputs = rollouts.obs[-1]
        rnn_hxs = rollouts.actor_rnn_states[-1]
        masks = rollouts.masks[-1]

        inputs = self._embed_if_needed(obs_inputs)
        value, _, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, obs_inputs, rnn_hxs, masks, action):
        inputs = self._embed_if_needed(obs_inputs)
        value, actor_features, critic_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

# Asymmetric Actor Critic: Critic can take in states
class A2CAsym(nn.Module):
    def __init__(self, config, actor_recurrent, use_embedding, embed_size=128):
        super(A2CAsym, self).__init__()
        self.use_embedding = use_embedding
        action_size = config['action_size']

        if use_embedding:
            state_size = config['state_size']
            obs_size = config['obs_size']

            self.action_embed = nn.Embedding(action_size, embed_size)
            self.obs_embed = nn.Embedding(obs_size, embed_size)
            self.state_embed = nn.Embedding(state_size, embed_size)

            actor_input_size = 2 * embed_size
            critic_input_size = embed_size
            self.embed_size = embed_size

        else:
            actor_input_size = config['obs_dim']
            critic_input_size = config['state_dim']

        self.base = MLPBaseAsym(actor_input_size, critic_input_size, actor_recurrent)
        self.dist = Categorical(self.base.output_size, action_size)

    @property
    def rnn_state_size(self):
        """Size of rnn_hx."""
        return self.base.rnn_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _embed_if_needed(self, obs, states):
        if not self.use_embedding:
            return obs, states
        action_embedded = self.action_embed(obs[:, 1])
        obs_embedded = self.obs_embed(obs[:, 0])
        return (bind(action_embedded, obs_embedded).reshape(-1, 2 * self.embed_size),
                self.state_embed(states).reshape(-1, self.embed_size))

    def act(self, rollouts, step, args, deterministic=False):
        obs_inputs = rollouts.obs[step]
        state_inputs = rollouts.state[step]
        rnn_hxs = rollouts.actor_rnn_states[step]
        masks = rollouts.masks[step]

        inputs_actor, inputs_critic = self._embed_if_needed(obs_inputs, state_inputs)
        value, actor_features, rnn_hxs = self.base(inputs_actor, inputs_critic, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs, None

    def get_value(self, rollouts, args):
        obs_inputs = rollouts.obs[-1]
        state_inputs = rollouts.state[-1]
        rnn_hxs = rollouts.actor_rnn_states[-1]
        masks = rollouts.masks[-1]

        inputs_actor, inputs_critic = self._embed_if_needed(obs_inputs, state_inputs)
        value, _, _ = self.base(inputs_actor, inputs_critic, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs_actor, inputs_critic, rnn_hxs, masks, action):
        inputs_actor, inputs_critic = self._embed_if_needed(inputs_actor, inputs_critic)
        value, actor_features, rnn_hxs = self.base(inputs_actor, inputs_critic, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy