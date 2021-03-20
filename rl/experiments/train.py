from rl.algo import Agent
from rl.algo import TestAgent
from rl.nn.envs import make_vec_envs

import time
import numpy as np
import random

import wandb

class TrainEx:
    def __init__(self, args, log_dir):
        self.args = args
        print("Make env training")
        self.train_envs = make_vec_envs(self.args, allow_early_resets=False)
        self.train_agent = Agent(self.train_envs, self.args, log_dir)

    def _train(self, n_update, num_updates):
        self.train_agent.train(n_update, num_updates)

    def _save(self):
        self.train_agent.save()

    def _evaluate(self):
        print("Make env testing")
        self.test_envs = make_vec_envs(self.args, allow_early_resets=True, seed_change=random.randint(0, 1000))
        self.test_agent = TestAgent(self.test_envs, self.args, self.train_agent.actor_critic.rnn_state_size)
        self.test_agent.evaluate(self.train_agent.actor_critic)

    def _print_train_stat(self, n_update, timestep, start, end, log_wandb):
        episode_rewards, value_loss, action_loss, dist_entropy = self.train_agent.get_statistic()
        update_str = "Updates {}, training timesteps {}, FPS {}".format(n_update, timestep, int(timestep / (end - start)))
        if len(episode_rewards) > 1:
            print(update_str)
            print("Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max {:.1f}/{:.1f}"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            print("Policy entropy: {:.3f}, Critic Loss: {:.3f}, Actor Loss {:.3f}\n".format(dist_entropy, value_loss, action_loss))    

            if log_wandb:
                wandb.log({'Num. Updates': n_update, 
                           'FPS':  int(timestep / (end - start)), 
                           'Average Return': np.mean(episode_rewards)}, 
                            step=timestep)        

    def _print_test_stat(self, update_str):
        episode_rewards = self.test_agent.get_statistic()
        if len(episode_rewards) > 1:
            print(update_str)
            print("Last {} testing episodes: mean/median reward {:.2f}/{:.2f}, min/max {:.1f}/{:.1f}\n"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

    def run(self, num_updates, start):
        if self.args.wandb:
            wandb.init(project='BGN-Test', group=self.args.group,
               settings=wandb.Settings(_disable_stats=True),
               name='_'.join([self.args.algo, self.args.env_name, 'seed=' + str(self.args.seed)]))

        for n_update in range(num_updates):

            self._train(n_update, num_updates)
        
            if (n_update % self.args.save_interval == 0 or n_update == num_updates - 1) and self.args.save_dir != "":
                self._save()

            if n_update % self.args.log_interval == 0 and n_update > 0:
                total_num_steps = (n_update + 1) * self.args.num_processes * self.args.num_steps
                end = time.time()
                
                self._print_train_stat(n_update, total_num_steps, start, end, self.args.wandb)

            if (self.args.eval_interval is not None and n_update % self.args.eval_interval == 0 and n_update > 0):
                total_num_steps = (n_update + 1) * self.args.num_processes * self.args.num_steps
                end = time.time()
                status_str = "Updates {}, testing timesteps {}, FPS {}".format(n_update, total_num_steps, int(total_num_steps / (end - start)))
                self._evaluate()
                self._print_test_stat(status_str)  