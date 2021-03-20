import gym
import gym_pomdps

import numpy as np
import bz2
import dill
import torch

from rl.algo.agent_heuristic import HeuristicAgent
from rl.parsing.arguments import get_args
from rl.storage import RolloutStorage
from rl.misc.utils import env_config
from rl.nn.envs import make_vec_envs

args = get_args()

experience_mem = []

config = env_config(args.env_name)
envs = make_vec_envs(args, allow_early_resets=True)
rollouts = RolloutStorage(args, config, 256, True)

episode_rewards = []

agent = HeuristicAgent()

envs.reset()

obs = envs.reset()
state = envs.get_state()
belief = envs.get_belief()

rollouts.obs[0].copy_(obs)
rollouts.state[0].copy_(state)
rollouts.belief[0].copy_(belief)

for _ in range(1000):
    for step in range(args.num_steps):
        # Sample actions
        _states = envs.get_state().numpy()
        _beliefs = envs.get_belief().numpy()

        action = torch.zeros((args.num_processes, 1), dtype=torch.int32)

        for i in range(args.num_processes):

            _state = int(_states[i, 0])
            _belief = _beliefs[i, :]

            action[i, :] = agent._compute_action(_state, _belief)
        
        actor_hidden = torch.zeros((args.num_processes, 256))
        critic_hidden = torch.zeros((args.num_processes, 256))
        action_log_probs = torch.zeros((args.num_processes, 1))

        # Observe reward and next obs
        obs, reward, done, infos = envs.step(action)

        state_ts = torch.empty((args.num_processes, config['state_dim']), dtype=torch.float)
        belief_ts = torch.empty((args.num_processes, config['belief_dim']), dtype=torch.float)
        index = 0

        for info in infos:
            state_ts[index] = torch.FloatTensor(info['curr_state'])
            belief_ts[index] = torch.FloatTensor(info['belief'])
            index += 1

            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, state_ts, belief_ts, actor_hidden, critic_hidden, action,
                    action_log_probs, torch.FloatTensor([0.0]), reward, masks, bad_masks)

    with torch.no_grad():
        experience_mem.append([rollouts.obs.clone().numpy(), 
                                rollouts.masks.clone().numpy(), 
                                rollouts.belief.clone().numpy(), 
                                rollouts.actions.clone().numpy(), 
                                rollouts.state.clone().numpy(),
                                ])

    rollouts.after_update()

print("Saving ...")
sfile = bz2.BZ2File("test.exp", "wb")
dill.dump(experience_mem, sfile)  
print("Done") 