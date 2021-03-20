import gym
import time
import warnings

from rl.parsing.arguments import get_args
from rl.experiments import ExperimentManager

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    args = get_args()

    experiment_manager = ExperimentManager(args)

    start_time = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    experiment_manager.run(num_updates, start_time)

if __name__ == "__main__":
    main()
