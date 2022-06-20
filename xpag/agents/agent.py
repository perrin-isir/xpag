# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple
import numpy as np
import jax.numpy as jnp


class Agent(ABC):
    def __init__(
        self,
        name: str,
        observation_dim: int,
        action_dim: int,
        params: Union[None, dict],
    ):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.params = params

    @abstractmethod
    def train_on_batch(
        self,
        batch: Dict[str, Union[np.ndarray, jnp.ndarray]],
    ) -> dict:
        pass

    @abstractmethod
    def select_action(
        self,
        observation: Union[np.ndarray, jnp.ndarray],
        eval_mode=False,
    ) -> Union[np.ndarray, jnp.ndarray, Tuple[Union[np.ndarray, jnp.ndarray], Dict]]:
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
