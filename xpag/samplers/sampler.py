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


class DefaultSampler:
    def __init__(self, version: str = 'torch'):
        self.version = version

    def sample(self, buffers, batch_size_in_transitions):
        # t_max = buffers['actions'].shape[1]
        rollout_batch_size = buffers[list(buffers.keys())[0]].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_max_episodes = buffers['episode_length'][episode_idxs, 0].flatten()
        if self.version == 'torch':
            t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
        else:
            t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        return transitions
