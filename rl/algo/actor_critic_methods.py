from abc import ABC, abstractmethod
import torch


class ActorCriticMethod(ABC):
    def __init__(self, actor_critic, args, config):
        self.actor_critic = actor_critic
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.belief_dim = config['belief_dim']
        self.state_dim = config['state_dim']
        self.n_known_states = config['n_known_states']
        self.belief_loss_coef = args.belief_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.optimizer = torch.optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

    def update_entropy_coef(self, value):
        self.entropy_coef = value

    def compute_cross_entropy(self, target, predict):
        return torch.mean(torch.sum(-target * torch.nn.LogSoftmax(dim=-1)(predict), dim=1))

    def update(self, rollouts):
        action_shape = rollouts.actions.size()[-1]

        num_steps, num_processes, _ = rollouts.rewards.size()

        inputs = self._prepare_inputs(rollouts)  # Subclass overrides this to prepare the various inputs

        try:
            values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(*inputs)
            belief_loss_enabled = False
        except ValueError:
            values, action_log_probs, dist_entropy, actor_belief_recon, critic_belief_recon = self.actor_critic.evaluate_actions(*inputs)
            belief_loss_enabled = True

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if belief_loss_enabled and self.belief_loss_coef > 0.0:
            true_beliefs = rollouts.belief[:-1, :, self.n_known_states:]
            actor_belief_recon, critic_belief_recon = map(lambda x: x.view(num_steps, num_processes, -1), [actor_belief_recon, critic_belief_recon])
            belief_loss = self.compute_cross_entropy(true_beliefs, actor_belief_recon) \
                          + self.compute_cross_entropy(true_beliefs, critic_belief_recon)

        self.optimizer.zero_grad()
        total_loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

        if belief_loss_enabled and self.belief_loss_coef > 0.0:
            total_loss += belief_loss * self.belief_loss_coef

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item() * self.value_loss_coef, action_loss.item(), dist_entropy.item()

    @abstractmethod
    def _prepare_inputs(self, rollouts):
        raise NotImplementedError


class ABCB(ActorCriticMethod):
    def _prepare_inputs(self, rollouts):
        action_shape = rollouts.actions.size()[-1]
        return (rollouts.belief[:-1].view(-1, self.belief_dim),
                rollouts.belief[:-1].view(-1, self.belief_dim),
                rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.critic_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))


class AHCB(ActorCriticMethod):
    def _prepare_inputs(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        return (rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.belief[:-1].view(-1, self.belief_dim),
                rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.critic_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))


class AHCH(ActorCriticMethod):
    def _prepare_inputs(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        return (rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))


class AHCS(ActorCriticMethod):
    def _prepare_inputs(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        return (rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.state[:-1].view(-1, self.state_dim),
                rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))


class ASCS(ActorCriticMethod):
    def _prepare_inputs(self, rollouts):
        action_shape = rollouts.actions.size()[-1]
        return (rollouts.state[:-1].view(-1, self.state_dim),
                rollouts.state[:-1].view(-1, self.state_dim),
                rollouts.actor_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.critic_rnn_states[0].view(-1, self.actor_critic.rnn_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))