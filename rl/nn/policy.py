import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.misc.distributions import Categorical
from rl.misc.utils import init, bind
from rl.nn.second_base import MLPBase, MLPBaseAsym, MLPBaseAsymDual
from rl.nn.clone_base import CloneBase

# Separating backbone between actor and critic
class A2C(nn.Module):
    def __init__(self, dims, n_reactive, recurrent, use_embedding, embed_size=128):
        super(A2C, self).__init__()
        self.n_reactive = n_reactive
        self.use_embedding = use_embedding

        if use_embedding:
            self.embed_size = embed_size
            self.action_embed = nn.Embedding(dims[1], embed_size)
            self.obs_embed = nn.Embedding(dims[0], embed_size)
            mlp_input_size = 2 * embed_size * n_reactive
        else:
            mlp_input_size = dims[0] * n_reactive

        self.base = MLPBase(mlp_input_size, recurrent=recurrent)
        self.dist = Categorical(self.base.output_size, dims[1])

        belief_size = dims[2] - dims[3]

        self.actor_belief_recon = nn.Linear(self.base.output_size, belief_size)
        self.critic_belief_recon = nn.Linear(self.base.output_size, belief_size)

    @property
    def rnn_state_size(self):
        """Size of rnn_hx."""
        return self.base.rnn_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _embed_if_needed(self, obs):
        if not self.use_embedding:
            return obs

        # The past `n_reactive` observations have already been concatenated together, but
        # we don't want the embedding layer to treat them as a single mega-observation.
        # So we separate them here, and then concatenate them again after the embedding.
        # Note the shape is unaffected when n=1, which is exactly the behavior we want.

        n = self.n_reactive
        obs = obs.view(obs.shape[0] * n, -1)        
        actions_embedded = self.action_embed(obs[:, 1])
        obs_embedded = self.obs_embed(obs[:, 0])
        return bind(actions_embedded, obs_embedded).reshape(-1, 2 * self.embed_size * n)

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

        actor_belief_recon = self.actor_belief_recon(actor_features)
        critic_belief_recon = self.critic_belief_recon(critic_features)

        return value, action_log_probs, dist_entropy, actor_belief_recon, critic_belief_recon

# Asymmetric Actor Critic: Critic can take in either beliefs or states
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

# Naive Behavior Cloning
class BC(nn.Module):
    def __init__(self, config, use_embedding, hidden_size=256, embed_size=128):
        super(BC, self).__init__()
        self.use_embedding = use_embedding
        self.hidden_size = hidden_size

        if self.use_embedding:
            action_size = config['action_size']
            obs_size = config['obs_size']

            self.embed_size = embed_size
            input_size = 2 * embed_size

            self.obs_embed = nn.Embedding(obs_size, embed_size)
            self.action_embed = nn.Embedding(action_size, embed_size)

        else:
            input_size = config['obs_dim']
            action_size = config['action_size']

        self.base = CloneBase(input_size, action_size, hidden_size=hidden_size)
        self.dist = Categorical(self.base.output_size, action_size)

    @property
    def rnn_state_size(self):
        return self.hidden_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _embed_if_needed(self, obs):
        if not self.use_embedding:
            return obs
        actions_embedded = self.action_embed(obs[:, 1])
        obs_embedded = self.obs_embed(obs[:, 0])
        return bind(actions_embedded, obs_embedded).reshape(-1, 2 * self.embed_size)

    def compute_alogits(self, inputs_actor, masks, num_processes, deterministic=False):
        inputs_actor = self._embed_if_needed(inputs_actor)
        rnn_hxs = torch.zeros((num_processes, self.hidden_size))
        alogits, rnn_hxs = self.base(inputs_actor, rnn_hxs, masks)
        dist = self.dist.distribution(alogits)
        return dist, rnn_hxs

    def act(self, rollouts, step, deterministic=False):
        inputs_actor = self._embed_if_needed(rollouts.obs[step])
        rnn_hxs = rollouts.actor_rnn_states[step]
        masks = rollouts.masks[step]

        alogits, rnn_hxs = self.base(inputs_actor, rnn_hxs, masks)

        dist = self.dist(alogits)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, rnn_hxs

class A2CAsymDual(nn.Module):
    def __init__(self, dims, critic_recurrent, actor_recurrent, use_embedding, embed_size=128):
        super(A2CAsymDual, self).__init__()
        self.use_embedding = use_embedding
        action_size = dims[2]

        if use_embedding:
            self.embed_size = embed_size
            self.action_embed = nn.Embedding(dims[2], embed_size)
            self.obs_embed = nn.Embedding(dims[1], embed_size)

            actor_input_dim = 2 * embed_size
            critic_input_dim = dims[0]
        else:
            actor_input_dim = dims[0]
            critic_input_dim = dims[1]

        self.base = MLPBaseAsymDual(actor_input_dim, critic_input_dim, critic_recurrent, actor_recurrent)
        self.dist = Categorical(self.base.output_size, action_size)

        self.actor_belief_recon = nn.Linear(self.base.output_size, dims[0] - dims[3])
        self.critic_belief_recon = nn.Linear(self.base.output_size, dims[0] - dims[3])

    @property
    def rnn_state_size(self):
        """Size of rnn_hx."""
        return self.base.rnn_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act_simple(self, rollouts, step, args, deterministic=False):
        if self.use_embedding:
            raise NotImplementedError

        actor_rnn_hxs = torch.zeros(args.num_processes, self.base.rnn_state_size)
        critic_rnn_hxs = torch.zeros(args.num_processes, self.base.rnn_state_size)
        inputs_actor = torch.FloatTensor(rollouts[2][step])
        inputs_critic = torch.FloatTensor(rollouts[2][step])
        masks = torch.FloatTensor(rollouts[1][step])

        value, actor_features, _, actor_rnn_hxs, critic_rnn_hxs = self.base(inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return action

    def _embed_if_needed(self, obs):
        if not self.use_embedding:
            return obs
        actions_embedded = self.action_embed(obs[:, 1])
        obs_embedded = self.obs_embed(obs[:, 0])
        return bind(actions_embedded, obs_embedded).reshape(-1, 2 * self.embed_size)

    def act(self, rollouts, step, args, deterministic=False):
        if self.use_embedding:
            inputs_actor = self._embed_if_needed(rollouts.obs[step])
            inputs_critic = rollouts.belief[step]
        else:
            if args.algo in ['ab-cb', 'bc']:
                inputs_actor = rollouts.belief[step]
                inputs_critic = rollouts.belief[step]

            if args.algo in ['ah-cb']:
                inputs_actor = rollouts.obs[step]
                inputs_critic = rollouts.belief[step]

            if args.algo in ['ah-chs']:
                inputs_actor = rollouts.obs[step]
                inputs_critic = rollouts.state[step]

            if args.algo in ['ah-csb']:
                inputs_actor = rollouts.obs[step]
                inputs_critic = bind(rollouts.state[step], rollouts.belief[step])

        actor_rnn_hxs = rollouts.actor_rnn_states[step]
        critic_rnn_hxs = rollouts.critic_rnn_states[step]
        masks = rollouts.masks[step]

        value, actor_features, _, actor_rnn_hxs, critic_rnn_hxs = self.base(inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, actor_rnn_hxs, critic_rnn_hxs

    def get_value(self, rollouts, args):
        if self.use_embedding:
            inputs_actor = self._embed_if_needed(rollouts.obs[-1])
            inputs_critic = rollouts.belief[-1]
        else:
            if args.algo in ['ab-cb']:
                inputs_actor = rollouts.belief[-1]
                inputs_critic = rollouts.belief[-1]

            if args.algo in ['ah-cb']:
                inputs_actor = rollouts.obs[-1]
                inputs_critic = rollouts.belief[-1]

            if args.algo in ['ah-chs']:
                inputs_actor = rollouts.obs[-1]
                inputs_critic = rollouts.state[-1]

            if args.algo in ['ah-csb']:
                inputs_actor = rollouts.obs[-1]
                inputs_critic = bind(rollouts.state[-1], rollouts.belief[-1])

        actor_rnn_hxs = rollouts.actor_rnn_states[-1]
        critic_rnn_hxs = rollouts.critic_rnn_states[-1]
        masks = rollouts.masks[-1]

        value, _, _, _, _ = self.base(inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks, action):
        inputs_actor = self._embed_if_needed(inputs_actor)
        value, actor_features, critic_features, actor_rnn_hxs, critic_rnn_hxs = self.base(inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        actor_belief_recon = self.actor_belief_recon(actor_features)
        critic_belief_recon = self.critic_belief_recon(critic_features)

        return value, action_log_probs, dist_entropy, actor_belief_recon, critic_belief_recon
