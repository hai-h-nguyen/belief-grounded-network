from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import sys
import pickle

class RolloutMemory(Dataset):

    def __init__(self, args):
        self.rollouts = []

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        current_len = len(self.rollouts)

        assert(current_len > 0)
        assert(idx < current_len)

        self.rollouts[idx]

        # return [self.rollouts[idx].obs, self.rollouts[idx].masks, self.rollouts[idx].actions]

    def getRollout(self, idx):
        return self.rollouts[idx]