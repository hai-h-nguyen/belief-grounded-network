import torch

class RolloutStorage(object):
    def __init__(self, args, config, rnn_hidden_dim, embed=False):

        num_steps = args.num_steps
        num_processes = args.num_processes

        self.obs = torch.zeros(num_steps + 1, num_processes, args.n_reactive * config['obs_dim'])

        self.state = torch.zeros(num_steps + 1, num_processes, config['state_dim'])
        self.belief = torch.zeros(num_steps + 1, num_processes, config['belief_dim'])

        if embed:
            self.obs = self.obs.long()
            self.state = self.state.long()

        self.actor_rnn_states = torch.zeros(num_steps + 1, num_processes, rnn_hidden_dim)

        self.critic_rnn_states = torch.zeros(num_steps + 1, num_processes, rnn_hidden_dim)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        
        self.actions = torch.zeros(num_steps, num_processes, 1)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)

        self.state = self.state.to(device)
        self.belief = self.belief.to(device)

        self.actor_rnn_states = self.actor_rnn_states.to(device)
        self.critic_rnn_states = self.critic_rnn_states.to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, state, belief, actor_hidden_states, critic_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):

        self.obs[self.step + 1].copy_(obs)

        self.state[self.step + 1].copy_(state)
        self.belief[self.step + 1].copy_(belief)

        self.actor_rnn_states[self.step + 1].copy_(actor_hidden_states)

        if critic_hidden_states is not None:
            self.critic_rnn_states[self.step + 1].copy_(critic_hidden_states)

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def insert_minimal(self, obs, actor_hidden_states, actions, masks):
        self.obs[self.step + 1].copy_(obs)

        self.actor_rnn_states[self.step + 1].copy_(actor_hidden_states)

        self.actions[self.step].copy_(actions)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps        

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])

        self.state[0].copy_(self.state[-1])
        self.belief[0].copy_(self.belief[-1])

        self.actor_rnn_states[0].copy_(self.actor_rnn_states[-1])
        self.critic_rnn_states[0].copy_(self.critic_rnn_states[-1])

        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, args):
        self.value_preds[-1] = next_value
        gae = 0.0
        for i in reversed(range(self.rewards.size(0))):
            delta = self.rewards[i] + args.gamma * self.value_preds[i+1] * self.masks[i+1] - self.value_preds[i]
            gae = delta + args.gamma * args.gae_lambda * self.masks[i+1] * gae
            if args.use_proper_time_limits:
                gae *= self.bad_masks[i+1]
            self.returns[i] = gae + self.value_preds[i]