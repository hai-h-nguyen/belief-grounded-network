import glob
import os
import torch

def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def set_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def set_deterministic(cuda_deterministic):
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def bind(*inputs):
    return torch.cat(inputs, dim=-1)

def env_config(env_name):
    config = {}

    if  env_name in ['PomdpRs44-v0']: 
        config['belief_dim'] = 257
        config['n_known_states'] = 0
        config['obs_size'] = 2
        config['obs_dim'] = 2
        config['state_size'] = 257 + 1
        config['action_size'] = 9
        config['state_dim'] = 1
        config['discount'] = 0.95
        
        return config

    if  env_name in ['PomdpRs55-v0']:        
        config['belief_dim'] = 801
        config['n_known_states'] = 0
        config['obs_size'] = 2
        config['obs_dim'] = 2
        config['state_size'] = 801 + 1
        config['action_size'] = 10
        config['state_dim'] = 1
        config['discount'] = 0.95

        return config

    if  env_name in ['PomdpHallway2-v0']:
        config['belief_dim'] = 92
        config['n_known_states'] = 0
        config['obs_dim'] = 2
        config['obs_size'] = 17
        config['action_size'] = 5
        config['state_dim'] = 1
        config['discount'] = 0.95
        config['state_size'] = 92 + 1

        return config

    if env_name in ['PomdpHallway-v0']:
        config['belief_dim'] = 60
        config['n_known_states'] = 0
        config['obs_dim'] = 2
        config['state_size'] = 60 + 1
        config['obs_size'] = 21
        config['action_size'] = 5
        config['discount'] = 0.95
        config['state_dim'] = 1

        return config

    if env_name in ['MomdpBumps1d-v0']:
        config['belief_dim'] = 227
        config['n_known_states'] = 2
        config['obs_dim'] = 3
        config['obs_size'] = -1
        config['action_size'] = 4
        config['discount'] = 0.99
        config['state_dim'] = 4

        return config

    if env_name in ['MomdpBumps2d-v0']:
        config['belief_dim'] = 259
        config['n_known_states'] = 0
        config['obs_dim'] = 4
        config['state_size'] = 259 + 1
        config['obs_size'] = -1
        config['action_size'] = 5
        config['discount'] = 0.99
        config['state_dim'] = 7

        return config        

    if env_name in ['MomdpTopPlate-v0']:
        config['belief_dim'] = 10
        config['n_known_states'] = 0
        config['obs_dim'] = 3
        config['obs_size'] = -1
        config['action_size'] = 3
        config['discount'] = 0.99
        config['state_dim'] = 3

        return config               

    raise NameError('Unknown domain!')
