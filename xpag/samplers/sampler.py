# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
import numpy as np
import torch
from jaxlib.xla_extension import DeviceArray
from typing import Union, Dict
from xtmp.tools.utils import DataType


class Sampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(
        self,
        buffers: Dict[str, Union[torch.Tensor, np.ndarray, DeviceArray]],
        batch_size: int,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, DeviceArray]]:
        """Return a batch of transitions"""
        pass


class DefaultEpisodicSampler(Sampler):
    def __init__(self, datatype: DataType = DataType.NUMPY):
        assert (
            datatype == DataType.TORCH_CPU
            or datatype == DataType.TORCH_CUDA
            or datatype == DataType.NUMPY
        ), (
            "datatype must be DataType.TORCH_CPU, "
            "DataType.TORCH_CUDA or DataType.NUMPY."
        )
        self.datatype = datatype
        super().__init__()

    @staticmethod
    def sum(transitions) -> float:
        return sum([transitions[key].sum() for key in transitions.keys()])

    def sample(
        self,
        buffers: Dict[str, Union[torch.Tensor, np.ndarray]],
        batch_size: int,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        rollout_batch_size = buffers["episode_length"].shape[0]
        episode_idxs = np.random.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p=buffers["episode_length"][:, 0, 0]
            / buffers["episode_length"][:, 0, 0].sum(),
        )
        t_max_episodes = buffers["episode_length"][episode_idxs, 0].flatten()
        if self.datatype == DataType.TORCH_CPU or self.datatype == DataType.TORCH_CUDA:
            t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
        else:
            t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        return transitions
