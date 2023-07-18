import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from agents.sac.modules.utils import soft_update, hard_update
from agents.sac.modules.models import GaussianPolicy, QNetwork, DeterministicPolicy
from agents.sac.modules.buffer import *
from typing import Dict, List, Tuple
from utils.result_utils import *


class SAC(object):
    def __init__(self, args, env):
        self.num_inputs = env.observation_space.shape[0]
        # print(f"env shape: {env.observation_space.shape}")
        self.action_space = env.action_space.shape[1]
        # print(f"action shape: {env.action_space.shape}")
        self.algo_path = args.model_dir
        self.episode_sac = 0

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.env = env
        self.batch_size = args.batch_size
        self.initial_random_steps = args.initial_steps

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)

        self.critic_target = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        # self.memory = ReplayMemory(args.memory_size, args.seed)
        self.memory_size = args.memory_size
        self.memory = ReplayBuffer(self.num_inputs, self.action_space, self.memory_size, self.batch_size)
        # total steps count
        self.total_step = 0
        self.ep_step = 0
        self.episode = 0
        # mode: train / test
        self.is_test = False
        self.transition = list()

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr_actor)

            self.actor = GaussianPolicy(self.num_inputs, self.action_space, args.hidden_size, self.action_space).to(
                self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(self.num_inputs, self.action_space, args.hidden_size,
                                              self.action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def step(self, curr_obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, bool]:
        """Take an action and return the response of the env."""
        state_next, reward, done, info = self.env.step(action, self.ep_step)
        # print(f"reward: {reward}")
        if not self.is_test:
            self.memory.store(
                obs=curr_obs,
                act=action,
                rew=reward,
                next_obs=state_next,
                done=done
            )

        return state_next, reward, done, info

    def update_parameters(self, memory, updates):
        # Sample a batch from memory
        samples = memory.sample_batch()

        state_batch = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state_batch = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action_batch = torch.FloatTensor(samples["acts"]).to(self.device)
        reward_batch = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done_batch = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        mask_batch = 1 - done_batch
        # print(mask_batch)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target) * mask_batch

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.actor_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()

    def train(self, args):
        num_episode = args.max_episode
        max_step = args.max_step
        plotting_interval = args.plot_interval
        algo_name = str(num_episode) + "-" + str(max_step) + \
                    "-" + str(args.user_num) + "-" + str(args.pen_coeff) + \
                    "-" + args.drl_algo + "-" + str(args.batch_size) + \
                    "-" + args.ai_network
        """Train the agent."""
        list_results = []
        actor_losses = []
        critic_losses = []
        scores = []
        reward_list = []
        self.total_step = 0
        self.ep_step = 0

        for self.episode_sac in range(1, num_episode + 1):
            self.is_test = False
            state = self.env.reset()
            score = 0
            self.ep_step = 0

            for step in range(1, max_step + 1):
                self.total_step += 1
                self.ep_step += 1
                action = self.select_action(state)
                next_state, reward, done, info = self.step(curr_obs=state, action=action)
                state = next_state
                score += reward

                if done:
                    print(f"Done: Step {self.total_step} of episode: {self.episode_sac} have score {score}")
                    state = self.env.reset()
                    break

                if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                    actor_loss, critic_loss, policy_loss, ent_loss, alpha = self.update_parameters(self.memory,
                                                                                                   self.total_step)

                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    reward_list.append(reward)
                if self.total_step % plotting_interval == 0:
                    self._plot(
                        self.total_step,
                        reward_list,
                        actor_losses,
                        critic_losses,
                    )
                    pass
                scores.append(score)
            print(f"Episode: {self.episode_sac}|Round/Eps:{max_step}|"
                  f"Score {score}|Penalty:{self.env.penalty}|"
                  f"Trans Time:{sum(self.env.t_trans[0]) / len(self.env.t_trans[0])}|"
                  f"Energy:{self.env.E}|Iu:{self.env.num_Iu}|"
                  f"Global Round: {self.env.num_Iglob}")
        if args.save_flag:
            save_results(
                scores,
                actor_losses,
                critic_losses,
                reward_list,
                algo_name
            )
        save_item(item_actor=self.actor,
                  item_critic=self.critic,
                  item_name=algo_name,
                  folder_name=self.algo_path)
        df_results = pd.DataFrame(list_results, columns=['episode', 'score'])
        result_path = "./results/"
        file_path = result_path + "{}.csv".format(algo_name)
        df_results.to_csv(file_path)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(18, 3))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.savefig(fname="result.pdf")
        plt.show()
