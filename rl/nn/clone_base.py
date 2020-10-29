import torch
import torch.nn as nn

from rl.misc.utils import init
from rl.nn.first_base import NNBase

from rl.misc.distributions import Categorical

import numpy as np

class CloneBase(NNBase):
    def __init__(self, num_inputs, action_size, recurrent=True, hidden_size=256):
        super(CloneBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.recurrent = recurrent

        if recurrent:
            num_inputs = hidden_size

        self.hidden_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.train()

    @property
    def output_size(self):
        return self.hidden_size        

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        return hidden_actor, rnn_hxs