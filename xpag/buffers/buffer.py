import numpy as np
import torch
import gym
import string
import collections
import functools
import random
import time
import datetime
from IPython import embed

class DefaultBuffer:
    def __init__(self, dict_sizes: dict, episode_max_length: int, buffer_size: int,
                 version: str = 'torch', device: str = 'cpu'):
        self.version = version
        self.device = device
        self.T = episode_max_length
        self.size = int(buffer_size // self.T)
        self.current_size = 0
        self.buffers = {}
        self.dict_sizes = dict_sizes
        self.dict_sizes['episode_length'] = 1
        for key in dict_sizes:
            if self.version == 'torch':
                self.buffers[key] = torch.empty([self.size, self.T, dict_sizes[key]],
                                                device=self.device)
            else:
                self.buffers[key] = np.empty([self.size, self.T, dict_sizes[key]])

    def store_episode(self, num_envs: int, episode, episode_t: int):
        # idxs = self._get_storage_idx(inc=1)
        idxs = self._get_storage_idx(inc=num_envs)
        episode_dict = episode._asdict()
        if self.version == 'torch':
            ep_length = torch.full((num_envs, self.T, 1),
                                   float(episode_t),
                                   device=self.device)
        else:
            ep_length = np.full((num_envs, self.T, 1),
                                float(episode_t))
        for key in episode._fields:
            self.buffers[key][idxs][:episode_t, :] = episode_dict[key]
        self.buffers['episode_length'][idxs] = ep_length

    def pre_sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][: self.current_size]
        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
