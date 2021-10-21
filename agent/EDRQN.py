import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network.RecurrentCritic import RecurrentCritic
from exploration.EpsilonExploration import EpsilonExploration
from memory.FixedLenEpisodicReplayBuffer import FixedLenEpisodicReplayBuffer
from torch.distributions.categorical import Categorical
from utilities.util_function import stable_log_softmax
import random
import math


class EDRQN(nn.Module):
    def __init__(self, config, agent_idx):
        super(EDRQN, self).__init__()
        self.config = config
        self.agent_idx = agent_idx
        self.observation_size = config['obs_size']
        self.action_size = config['act_size']
        self.device = config['device']

        self.critic_local = RecurrentCritic(
            input_dim=[self.observation_size, 1],  # obs + target_entropy
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            output_dim=self.action_size
        ).to(self.device)  # type: RecurrentCritic
        self.critic_target = RecurrentCritic(
            input_dim=[self.observation_size, 1],  # obs + target_entropy
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            output_dim=self.action_size
        ).to(self.device)  # type: RecurrentCritic
        for to_model, from_model in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.alpha_local = RecurrentCritic(
            input_dim=[self.action_size, 1],  # act + target_entropy
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            output_dim=1
        ).to(self.device)  # type: RecurrentCritic
        self.alpha_target = RecurrentCritic(
            input_dim=[self.action_size, 1],  # act + target_entropy
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            output_dim=1
        ).to(self.device)  # type: RecurrentCritic
        for to_model, from_model in zip(self.alpha_target.parameters(), self.alpha_local.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=config['lr'], eps=1e-4)
        self.alpha_optim = optim.Adam(self.alpha_local.parameters(), lr=config['lr'], eps=1e-4)
        self.gamma = config['gamma']
        self.tau = config['tau']

        self.exploration = EpsilonExploration(epsilon=config['epsilon'], action_size=self.action_size)
        self.memory = FixedLenEpisodicReplayBuffer(
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            obs_size=self.observation_size,
            act_size=1,
            device=self.device,
            max_episode_len=config['mini_episode_len'],
            extend=True
        )
        self.reset()
        self.prev_hidden_critic = None
        self.prev_hidden_alpha = None
        self.log_alpha = None
        self.target_entropy = self._init_target_entropy()

    def reset(self):
        self.memory.reset()

    def update_learning_rate(self, new_lr):
        for optimizer in [self.critic_optim, self.alpha_optim]:
            for g in optimizer.param_groups:
                g['lr'] = new_lr

    def _init_target_entropy(self):
        return torch.tensor(
            random.uniform(1e-5, -math.log(1. / self.action_size, 2.) * self.config['entropy_scale']),
            dtype=torch.float32,
            device=self.device
        )

    def update_prev_hidden(self, done):
        if done:
            self.prev_hidden_critic = None
            self.prev_hidden_alpha = None
            self.target_entropy = self._init_target_entropy()

    def store_experience(self, obs, act, rew, next_obs, done):
        self.memory.store_experience(obs, act, rew, next_obs, done, self.target_entropy.cpu())

    def pick_action(self, obs, on_training):
        obs_ = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        self.critic_local.eval()
        with torch.no_grad():
            action_q_values, self.prev_hidden_critic = self.critic_local.get_q_value(
                [obs_, self.target_entropy.reshape([1, 1])], self.prev_hidden_critic
            )
            self.log_alpha, self.prev_hidden_alpha = self.alpha_local.get_q_value(
                [action_q_values, self.target_entropy.reshape([1, 1])], self.prev_hidden_alpha
            )
            action_log_prob = stable_log_softmax(action_q_values / torch.exp(self.log_alpha))
            dist = Categorical(probs=torch.exp(action_log_prob))
            action = dist.sample().item()
        self.critic_local.train()
        if on_training:
            action = self.exploration.perturb_action_with_annealing_noise(
                action,
                total_step_to_anneal=self.config['total_step_to_anneal'],
                current_step=self.config['current_global_step'],
                total_episode_to_anneal=self.config['total_episode_to_anneal'],
                current_episode=self.config['current_episode']
            )
        return action

    def learn(self):
        if self._time_to_learn():
            loss_critic, loss_alpha = self._compute_loss()
            self.critic_optim.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.config['max_norm'])
            self.critic_optim.step()

            self.alpha_optim.zero_grad()
            loss_alpha.backward()
            torch.nn.utils.clip_grad_norm_(self.alpha_local.parameters(), self.config['max_norm'])
            self.alpha_optim.step()

            for to_model, from_model in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                assert not torch.any(torch.isnan(from_model)), 'model nan error'
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
            for to_model, from_model in zip(self.alpha_target.parameters(), self.alpha_local.parameters()):
                assert not torch.any(torch.isnan(from_model)), 'model nan error'
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)

    def _time_to_learn(self):
        return self.memory.current_size > self.config['batch_size']

    def _compute_loss(self):
        ep_obs, ep_act, ep_rew, ep_next_obs, ep_done, ep_ent = self.memory.sample_experience(with_extend=True)
        # A full episode ends when the ep_done turns to True, and the following mask ensures that the last step is involved
        mask = torch.cat([torch.ones((self.config['batch_size'], 1), device=self.device, dtype=torch.bool), ~ep_done[:, :-1, 0]], dim=1)

        hidden_critic_eval = None
        hidden_critic_target = None
        hidden_alpha_eval = None
        hidden_alpha_target = None

        for step_idx in range(self.config['mini_episode_len']):
            # calculate q_target
            with torch.no_grad():
                q_values_target_next, hidden_critic_target = self.critic_target.get_q_value(
                    [ep_next_obs[:, step_idx, :], ep_ent[:, step_idx, :]],
                    hidden_critic_target
                )
                log_alpha_next, hidden_alpha_target = self.alpha_target.get_q_value(
                    [q_values_target_next, ep_ent[:, step_idx, :]],
                    hidden_alpha_target
                )
                act_next_log_prob = stable_log_softmax(q_values_target_next / torch.exp(log_alpha_next))
                act_next_prob = torch.exp(act_next_log_prob)
                # for the derivation of v_next, please refer to "Soft Actor-Critic Algorithms and Applications" by Haarnoja et al.
                v_next = torch.sum(
                    q_values_target_next * act_next_prob - torch.exp(log_alpha_next) * act_next_prob * act_next_log_prob,
                    dim=1, keepdim=True
                )
                q_target = ep_rew[:, step_idx, :] + self.gamma * v_next * (~ep_done[:, step_idx, :])

            # calculate q_expected
            q_values_expected, hidden_critic_eval = self.critic_local.get_q_value(
                [ep_obs[:, step_idx, :], ep_ent[:, step_idx, :]],
                hidden_critic_eval
            )
            log_alpha, hidden_alpha_eval = self.alpha_local.get_q_value(
                [q_values_expected.detach(), ep_ent[:, step_idx, :]],
                hidden_alpha_eval
            )
            q_expected = q_values_expected.gather(dim=1, index=ep_act[:, step_idx, :].long())

            act_log_prob = stable_log_softmax(q_values_expected.detach() / torch.exp(log_alpha))
            act_entropy = -torch.sum(torch.exp(act_log_prob) * act_log_prob, dim=1)

        # calculate td error
        loss_critic = F.mse_loss(q_expected * mask[:, self.config['mini_episode_len'] - 1],
                                 q_target * mask[:, self.config['mini_episode_len'] - 1])
        loss_alpha = torch.mean((act_entropy - self.target_entropy) ** 2)
        assert not torch.isnan(loss_critic), 'nan loss_critic error in step {}'.format(step_idx)
        assert not torch.isnan(loss_alpha), 'nan loss_alpha error in step {}'.format(step_idx)

        if self.config['save_result']:
            self.config['summary_writer'].add_scalar(
                'agent/EDRQN_changing_ent_{}_loss_critic'.format(self.agent_idx),
                loss_critic,
                self.config['current_episode']
            )
            self.config['summary_writer'].add_scalar(
                'agent/EDRQN_changing_ent_{}_loss_alpha'.format(self.agent_idx),
                loss_alpha,
                self.config['current_episode']
            )
        return loss_critic, loss_alpha
