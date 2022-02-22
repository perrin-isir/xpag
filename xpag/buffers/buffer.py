# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, Dict, NamedTuple
from xpag.tools.utils import DataType


class Buffer(ABC):
    def __init__(
        self,
        episode_max_length: int,
        buffer_size: int,
        datatype: DataType = DataType.TORCH,
        device: str = "cpu",
    ):
        assert datatype == DataType.TORCH or datatype == DataType.NUMPY
        self.T = episode_max_length
        self.buffer_size = buffer_size
        self.datatype = datatype
        self.device = device
        self.size = int(buffer_size // self.T)
        self.current_size = 0
        self.buffers = {}

    @abstractmethod
    def store_episode(self, num_envs: int, episode: NamedTuple, episode_length: int):
        """Store one or several episodes in the buffer"""
        pass

    @abstractmethod
    def pre_sample(self) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Return a part of the buffer from which the sampler will extract samples"""
        pass


class DefaultBuffer(Buffer):
    def __init__(
        self,
        dict_sizes: dict,
        episode_max_length: int,
        buffer_size: int,
        datatype: DataType,
        device: str = "cpu",
    ):
        super().__init__(episode_max_length, buffer_size, datatype, device)
        self.dict_sizes = dict_sizes
        self.dict_sizes["episode_length"] = 1
        for key in dict_sizes:
            if self.datatype == DataType.TORCH:
                self.buffers[key] = torch.empty(
                    [self.size, self.T, dict_sizes[key]], device=self.device
                )
            else:
                self.buffers[key] = np.empty([self.size, self.T, dict_sizes[key]])

    def store_episode(self, num_envs: int, episode: NamedTuple, episode_length: int):
        idxs = self._get_storage_idx(inc=num_envs)
        episode_dict = episode._asdict()
        if self.datatype == DataType.TORCH:
            ep_length = torch.full(
                (num_envs, self.T, 1), float(episode_length), device=self.device
            )
        else:
            ep_length = np.full((num_envs, self.T, 1), float(episode_length))
        for key in episode._fields:
            self.buffers[key][idxs, :episode_length, :] = episode_dict[key][
                :, :episode_length, :
            ]
        self.buffers["episode_length"][idxs] = ep_length

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
