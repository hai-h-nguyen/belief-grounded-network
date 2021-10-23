import gym
import time
import warnings

from rl.parsing.arguments import get_args
from rl.experiments import ExperimentManager

import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    args = get_args()

    experiment_manager = ExperimentManager(args)

    start_time = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    env_name = args.env_name[:-2]

    if args.running_mode in ['train']:
        wandb.init(project=env_name + 'exploration', settings=wandb.Settings(_disable_stats=True),
                    entity='hainh22',
                    group=args.group if args.group is not None else str(args.algo),
                    name='s' + str(args.seed))

    experiment_manager.run(num_updates, start_time)

if __name__ == "__main__":
    main()