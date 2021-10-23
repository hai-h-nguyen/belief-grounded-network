import os

import gym
import numpy as np
import torch
import collections

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from rl.domains import *

def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        return env

    return _thunk

def make_vec_envs(args, allow_early_resets, seed_change=0):
    envs = [
        make_env(args.env_name, args.seed + seed_change, i, args.log_dir, allow_early_resets)
        for i in range(0, args.num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs)

    envs = VecPyTorch(envs, args.device)

    return envs

def should_embed(env_name):
    # Classic POMDPs have discrete observations/actions that need to be embedded.
    # For brevity, we list the environments that do NOT need an embedding here.
    return not('Momdp' in env_name)

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def get_state(self):
        state = self.venv.get_state()
        state = np.vstack(state)
        state = torch.from_numpy(state).float().to(self.device)
        return state

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class HistoryWrapper(gym.Wrapper):
    '''Stacks the previous `history_len` observations along the first axis.
    Duplicates the first observation at the start of a new episode.'''
    def __init__(self, env, history_len):
        assert history_len > 1
        super().__init__(env)
        self.history_len = history_len
        self.deque = collections.deque(maxlen=history_len)

        self.shape = self.observation_space.shape
        self.observation_space.shape = (history_len * self.shape[0], *self.shape[1:])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.deque.append(observation)
        return self._history(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.history_len):
            self.deque.append(observation)
        return self._history()

    def _history(self):
        return np.concatenate(list(self.deque), axis=0)