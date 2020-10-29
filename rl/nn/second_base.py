import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.misc.utils import init
from rl.nn.first_base import NNBase, NNDualBase
import numpy as np

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.recurrent = recurrent

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, hidden_critic, rnn_hxs
        

class MLPBaseAsym(NNBase):
    def __init__(self, num_actor_inputs, num_critic_inputs, actor_recurrent, hidden_size=256):
        super(MLPBaseAsym, self).__init__(actor_recurrent, num_actor_inputs, hidden_size)

        self.actor_recurrent = actor_recurrent

        if actor_recurrent:
            num_actor_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_actor_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_critic_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs_actor, inputs_critic, rnn_hxs, masks):
        x = inputs_actor

        if self.actor_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(inputs_critic)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBaseAsymDual(NNDualBase):
    def __init__(self, num_actor_inputs, num_critic_inputs, critic_recurrent, actor_recurrent, hidden_size=256):
        super(MLPBaseAsymDual, self).__init__(num_actor_inputs, num_critic_inputs, hidden_size)

        if actor_recurrent:
            num_actor_inputs = hidden_size

        if critic_recurrent:
            num_critic_inputs = hidden_size

        self.critic_recurrent = critic_recurrent
        self.actor_recurrent = actor_recurrent

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_actor_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_critic_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs_actor, inputs_critic, actor_rnn_hxs, critic_rnn_hxs, masks):
        x = inputs_actor

        if self.actor_recurrent:
            x, actor_rnn_hxs = self._forward_gru_actor(x, actor_rnn_hxs, masks)

        y = inputs_critic

        if self.critic_recurrent:
            y, critic_rnn_hxs = self._forward_gru_critic(y, critic_rnn_hxs, masks)

        hidden_critic = self.critic(y)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, hidden_critic, actor_rnn_hxs, critic_rnn_hxs 