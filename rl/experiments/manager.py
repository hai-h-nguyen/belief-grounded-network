from rl.misc import utils
from .simulate import SimulateEx
from .train import TrainEx

import os
import torch

class ExperimentManager:
    def __init__(self, args):
        self.args = args

        self.device = self.args.device

        log_dir = os.path.expanduser(args.log_dir)
        utils.cleanup_log_dir(log_dir)        

        if (args.save_dir != ""):
            self.log_dir = os.path.join(args.save_dir, self.args.env_name)
            try:
                os.makedirs(self.log_dir)
            except OSError:
                pass

        utils.set_deterministic(args.cuda_deterministic)
        utils.set_randomness(args.seed)
        torch.set_num_threads(1)
        
    def run(self, num_updates, start_time):
        if (self.args.running_mode == 'simulate'):
            experiment = SimulateEx(self.args)
            experiment.run(num_updates)

        if (self.args.running_mode == 'random'):
            experiment = SimulateEx(self.args, random_agent=True)
            experiment.run(num_updates)

        if (self.args.running_mode == 'train'):
            experiment = TrainEx(self.args, self.log_dir)
            experiment.run(num_updates, start_time)                


