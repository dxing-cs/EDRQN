import numpy as np
from memory.EpisodicReplayBuffer import EpisodicReplayBuffer
from collections import deque


class FixedLenEpisodicReplayBuffer(EpisodicReplayBuffer):
    def __init__(self, buffer_size, batch_size, obs_size, act_size, device, max_episode_len, extend=False):
        super(FixedLenEpisodicReplayBuffer, self).__init__(buffer_size, batch_size, obs_size, act_size, device, max_episode_len, extend)
        self.obs_cache = deque(maxlen=max_episode_len)
        self.act_cache = deque(maxlen=max_episode_len)
        self.rew_cache = deque(maxlen=max_episode_len)
        self.next_obs_cache = deque(maxlen=max_episode_len)
        self.done_cache = deque(maxlen=max_episode_len)
        self.extend_cache = deque(maxlen=max_episode_len)

    def store_experience(self, obs, act, rew, next_obs, done, ext=None):
        if not done:
            self._allow_to_store = True

        if self._allow_to_store:
            self.obs_cache.append(obs)
            self.act_cache.append(act)
            self.rew_cache.append(rew)
            self.next_obs_cache.append(next_obs)
            self.done_cache.append(done)
            if ext:
                self.extend_cache.append(ext)

            if len(self.obs_cache) == self.max_episode_len or done:
                cache_len = len(self.obs_cache)
                self.memory['obs'][self._cur_buffer_idx][:cache_len] = np.array(self.obs_cache, dtype=np.float32).reshape((cache_len, -1))
                self.memory['act'][self._cur_buffer_idx][:cache_len] = np.array(self.act_cache, dtype=np.int64).reshape((cache_len, -1))
                self.memory['rew'][self._cur_buffer_idx][:cache_len] = np.array(self.rew_cache, dtype=np.float32).reshape((cache_len, -1))
                self.memory['next_obs'][self._cur_buffer_idx][:cache_len] = np.array(self.next_obs_cache, dtype=np.float32).reshape((cache_len, -1))
                self.memory['done'][self._cur_buffer_idx][:cache_len] = np.array(self.done_cache, dtype=np.bool).reshape((cache_len, -1))
                if ext:
                    self.memory['extend'][self._cur_buffer_idx][:cache_len] = np.array(self.extend_cache, dtype=np.float32).reshape((cache_len, -1))

                self.reward_stat['sum_rew'] += rew
                self.reward_stat['sum_rew_square'] += rew ** 2
                self.reward_stat['counter'] += 1

                self.current_size = min(self.current_size + 1, self.buffer_size)
                self._cur_buffer_idx = (self._cur_buffer_idx + 1) % self.buffer_size
                self.memory['act'][self._cur_buffer_idx] = np.zeros((self.max_episode_len, 1), dtype=np.int64)
                self.memory['rew'][self._cur_buffer_idx] = np.zeros((self.max_episode_len, 1), dtype=np.float32)
                self.memory['done'][self._cur_buffer_idx] = np.ones((self.max_episode_len, 1), dtype=np.bool)

            if done:
                self._clear_cache()
                self._allow_to_store = False

    def _clear_cache(self):
        self.obs_cache.clear()
        self.act_cache.clear()
        self.rew_cache.clear()
        self.next_obs_cache.clear()
        self.done_cache.clear()
        self.extend_cache.clear()
