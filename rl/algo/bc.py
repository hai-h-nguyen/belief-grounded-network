import torch.nn as nn
import torch.optim as optim
import torch

class BehaviorClone():
    def __init__(self, cloner, args, config):
        self.cloner = cloner
        self.max_grad_norm = args.max_grad_norm
        self.cloner_optim = optim.Adam(cloner.parameters(), args.bc_lr)

        self.batch_size = args.bc_batch_size
        self.num_processes = args.num_processes
        self.num_steps = args.num_steps

    def update(self, rollouts):
        obs = torch.transpose(rollouts[0], 1, 0).reshape(self.num_steps + 1, self.num_processes * self.batch_size, -1)
        masks = torch.transpose(rollouts[1], 1, 0).reshape(self.num_steps + 1, self.num_processes * self.batch_size, 1)
        actions = torch.transpose(rollouts[3], 1, 0).reshape(self.num_steps, self.num_processes * self.batch_size, 1)

        obs_shape = obs.size()[2:]

        alogits, _ = self.cloner.compute_alogits(obs[:-1].view(-1, *obs_shape), 
                                                        masks[:-1].view(-1, 1), self.num_processes * self.batch_size)
        self.cloner_optim.zero_grad()
        loss = nn.CrossEntropyLoss()(alogits, actions.reshape(self.num_processes * self.batch_size * self.num_steps, ))
        loss.backward()
        nn.utils.clip_grad_norm_(self.cloner.parameters(), self.max_grad_norm)
        self.cloner_optim.step()
        return loss.item()