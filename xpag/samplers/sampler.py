from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, Dict
from enum import Enum
import xpag


class DataType(Enum):
    TORCH = 'data represented as torch tensors'
    NUMPY = 'data represented as numpy arrays'


class Sampler(ABC):
    def __init__(self,
                 datatype: DataType = xpag.tl.DataType.TORCH):
        self.datatype = datatype

    @abstractmethod
    def sample(self,
               buffers: Dict[str, Union[torch.Tensor, np.ndarray]],
               batch_size_in_transitions: int
               ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Return a batch of transitions
        """
        pass


class DefaultSampler(Sampler):
    def __init__(self, datatype: DataType = DataType.TORCH):
        super().__init__(datatype)

    def sample(self,
               buffers : Dict[str, Union[torch.Tensor, np.ndarray]],
               batch_size_in_transitions: int):
        rollout_batch_size = buffers[list(buffers.keys())[0]].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_max_episodes = buffers['episode_length'][episode_idxs, 0].flatten()
        if self.datatype == DataType.TORCH:
            t_samples = (torch.rand_like(t_max_episodes) * t_max_episodes).long()
        else:
            t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffers[key][episode_idxs, t_samples] for key in buffers.keys()
        }
        return transitions
