import torch
import numpy as np
import math


class EpisodicReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_size, act_size, device, max_episode_len, extend=False):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.max_episode_len = max_episode_len

        self.memory = {
            'obs': np.empty([self.buffer_size, self.max_episode_len, self.obs_size], dtype=np.float32),
            'act': np.empty([self.buffer_size, self.max_episode_len, 1], dtype=np.int64),
            'rew': np.empty([self.buffer_size, self.max_episode_len, 1], dtype=np.float32),
            'next_obs': np.empty([self.buffer_size, self.max_episode_len, self.obs_size], dtype=np.float32),
            'done': np.empty([self.buffer_size, self.max_episode_len, 1], dtype=np.bool)
        }
        if extend:
            # for entropy term
            self.memory['extend'] = np.empty([self.buffer_size, self.max_episode_len, 1], dtype=np.float32)

        self._cur_buffer_idx = 0
        self.current_size = 0
        self.reward_stat = {
            'sum_rew': 0,
            'sum_rew_square': 0,
            'counter': 0,
        }
        self.step_in_episode = 0
        self._allow_to_store = True

    def reset(self):
        self._cur_buffer_idx = 0
        self.current_size = 0
        self.memory['act'][self._cur_buffer_idx] = np.zeros([self.max_episode_len, 1], dtype=np.int64)
        self.memory['rew'][self._cur_buffer_idx] = np.zeros([self.max_episode_len, 1], dtype=np.float32)
        self.memory['done'][self._cur_buffer_idx] = np.ones([self.max_episode_len, 1], dtype=np.bool)

    def store_experience(self, obs, act, rew, next_obs, done, ext=None):
        # Sometimes one agent has finished its episode but the other agent is still acting, and self._allow_to_store
        # ensures that only experiences before the first appearance of done=True are stored.
        if not done:
            self._allow_to_store = True

        if self._allow_to_store:
            self.memory['obs'][self._cur_buffer_idx][self.step_in_episode] = obs
            self.memory['act'][self._cur_buffer_idx][self.step_in_episode] = act
            self.memory['rew'][self._cur_buffer_idx][self.step_in_episode] = rew
            self.memory['next_obs'][self._cur_buffer_idx][self.step_in_episode] = next_obs
            self.memory['done'][self._cur_buffer_idx][self.step_in_episode] = done
            if ext is not None:
                self.memory['extend'][self._cur_buffer_idx][self.step_in_episode] = ext

            self.reward_stat['sum_rew'] += rew
            self.reward_stat['sum_rew_square'] += rew ** 2
            self.reward_stat['counter'] += 1
            self.step_in_episode += 1
            if done:
                self.current_size = min(self.current_size + 1, self.buffer_size)
                self._cur_buffer_idx = (self._cur_buffer_idx + 1) % self.buffer_size
                self.step_in_episode = 0
                self.memory['act'][self._cur_buffer_idx] = np.zeros([self.max_episode_len, 1], dtype=np.int64)
                self.memory['rew'][self._cur_buffer_idx] = np.zeros([self.max_episode_len, 1], dtype=np.float32)
                self.memory['done'][self._cur_buffer_idx] = np.ones([self.max_episode_len, 1], dtype=np.bool)
                self._allow_to_store = False

    def sample_experience(self, batch_size=None, norm_reward=False, with_extend=False):
        batch_size = batch_size or self.batch_size
        idxs = np.random.choice(self.current_size, batch_size, replace=False)

        obs = self.memory['obs'][idxs]
        act = self.memory['act'][idxs]
        rew = self.memory['rew'][idxs]
        next_obs = self.memory['next_obs'][idxs]
        done = self.memory['done'][idxs]
        if with_extend:
            extend = self.memory['extend'][idxs]

        if norm_reward:
            mean = self.reward_stat['sum_rew'] / self.reward_stat['counter']
            std = math.sqrt(self.reward_stat['sum_rew_square'] / self.reward_stat['counter'] - mean ** 2)
            rew = (rew - mean) / (abs(std) + 1e-6)

        if with_extend:
            return torch.from_numpy(obs).to(self.device), \
                   torch.from_numpy(act).to(self.device), \
                   torch.from_numpy(rew).to(self.device), \
                   torch.from_numpy(next_obs).to(self.device), \
                   torch.from_numpy(done).to(self.device), \
                   torch.from_numpy(extend).to(self.device)
        else:
            return torch.from_numpy(obs).to(self.device), \
                   torch.from_numpy(act).to(self.device), \
                   torch.from_numpy(rew).to(self.device), \
                   torch.from_numpy(next_obs).to(self.device), \
                   torch.from_numpy(done).to(self.device)
