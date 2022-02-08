from abc import ABC, abstractmethod
from typing import Union, Dict
import torch
import numpy as np
from xpag.samplers.sampler import Sampler


class Agent(ABC):
    def __init__(self, name: str, observation_dim: int, action_dim: int,
                 params: Union[None, dict]):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.params = params

    @abstractmethod
    def train(self,
              pre_sample: Dict[str, Union[torch.Tensor, np.ndarray]],
              sampler: Sampler,
              batch_size: int):
        pass

    @abstractmethod
    def select_action(self, observation: Union[torch.Tensor, np.ndarray],
                      deterministic=True):
        pass

    @abstractmethod
    def write_config(self, output_file: str):
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass
