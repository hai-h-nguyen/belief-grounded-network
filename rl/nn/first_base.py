import torch
import torch.nn as nn

from rl.misc.utils import init


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def rnn_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        return forward_gru(self.gru, x, hxs, masks)


class NNDualBase(nn.Module):
    def __init__(self, recurrent_actor_size, recurrent_critic_size, hidden_size):
        super(NNDualBase, self).__init__()

        self._hidden_size = hidden_size

        self.gru_actor = nn.GRU(recurrent_actor_size, hidden_size)
        for name, param in self.gru_actor.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.gru_critic = nn.GRU(recurrent_critic_size, hidden_size)
        for name, param in self.gru_critic.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    @property
    def rnn_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru_actor(self, x, hxs, masks):
        return forward_gru(self.gru_actor, x, hxs, masks)

    def _forward_gru_critic(self, x, hxs, masks):
        return forward_gru(self.gru_critic, x, hxs, masks)


def forward_gru(gru, x, hxs, masks):
    if x.size(0) == hxs.size(0):
        x, hxs = gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
        x = x.squeeze(0)
        hxs = hxs.squeeze(0)
    else:
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with masks
        masks = masks.view(T, N)

        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic cleaner
        has_zeros = ((masks[1:] == 0.0) \
                        .any(dim=-1)
                        .nonzero()
                        .squeeze()
                        .cpu())

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [T]

        hxs = hxs.unsqueeze(0)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hxs = gru(
                x[start_idx:end_idx],
                hxs * masks[start_idx].view(1, -1, 1))

            outputs.append(rnn_scores)

        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        # flatten
        x = x.view(T * N, -1)
        hxs = hxs.squeeze(0)

    return x, hxs
