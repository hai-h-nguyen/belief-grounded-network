import concurrent.futures
from subprocess import Popen, check_output
from argparse import ArgumentParser
import os
import yaml
from threading import Lock

def get_path(algo):
    if algo == 'ah-ch':
        return 'ahch'

    if algo == 'ah-cb':
        return 'ahcb'

    if algo == 'ah-cs':
        return 'ahcs'

    if algo == 'ab-cb':
        return 'abcb'

    if algo == 'bgn':
        return 'bgn'

    if algo == 'ab-cb-bgn':
        return 'abcb-bgn'

def run(cmd, path):
    with open(path, 'w') as f:
        p = Popen(cmd, stdout=f, stderr=f, shell=True)

    p.wait()

def make_filename(directory, algo, seed):
    filename = [directory] 
    filename += [get_path(algo)]
    filename += ['s' + str(seed) + '.txt']
    return '/'.join(filename)


def make_cmd(env, timesteps, algo, path, mode, seed):
    cmd  = ['python3 main.py']
    cmd += ['--env-name', env]
    cmd += ['--num-env-steps', str(timesteps)]
    cmd += ['--seed', str(seed)]

    if algo in ['bgn']:
        cmd += ['--algo', 'ah-ch']

    elif algo in ['ab-cb-bgn']:
        cmd += ['--algo', 'ab-cb']

    else:
        cmd += ['--algo', str(algo)]

    if algo in ['bgn']:
        cmd += ['--belief-loss-coef 1.0']

    if algo in ['ab-cb-bgn']:
        cmd += ['--belief-loss-coef 1.0']

    if env in ['PomdpRs44-v0', 'PomdpRs55-v0']:
        cmd += ['--use-linear-entropy-decay --entropy-coef 2.0']

    if mode in ['testing']:
        cmd += ['--eval-interval 50']

    return cmd


def make_joblist(experiments):
    jobs = []
    for exp in experiments:
        seed_min = exp['seed_min']
        seed_max = exp['seed_max']
        timesteps = exp['timesteps']
        directory = exp['directory']
        env = exp['env']
        mode = exp['mode']

        for algo in exp['algo']:
            for s in range(seed_min, seed_max + 1):
                filename = make_filename(directory, algo, seed=s)
                cmd = make_cmd(env, timesteps, algo, filename, mode, seed=s)
                jobs.append((cmd, filename))
                
    return jobs


def run_job(cmd, path):
    print(' '.join(cmd), flush=True)
    run(' '.join(cmd), path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target', type=str, help='Name of experiment to run. Use \'all\' to run everything.', default='hallway')
    parser.add_argument('--config', type=str, default='_automate.yaml')
    parser.add_argument('--n-workers', type=int, default=3)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)    

    experiments = config.values() if args.target == 'all' else [config[args.target]]
    jobs = make_joblist(experiments)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        for cmd, path in jobs:
            if os.path.exists(path):
                print(path, 'already exists', flush=True)
            else:
                executor.submit(run_job, cmd, path)
