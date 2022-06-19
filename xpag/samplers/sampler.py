# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from typing import Union, Dict


class Sampler(ABC):
    def __init__(self, *, seed: Union[int, None] = None):
        self.seed = seed
        pass

    @abstractmethod
    def sample(
        self,
        buffer,
        batch_size: int,
    ) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
        """Return a batch of transitions"""
        pass


class DefaultEpisodicSampler(Sampler):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sum(transitions) -> float:
        return sum([transitions[key].sum() for key in transitions.keys()])

    def sample(
        self,
        buffer: Dict[str, Union[np.ndarray]],
        batch_size: int,
    ) -> Dict[str, Union[np.ndarray]]:
        rollout_batch_size = buffer["episode_length"].shape[0]
        episode_idxs = np.random.choice(
            np.arange(rollout_batch_size),
            size=batch_size,
            replace=True,
            p=buffer["episode_length"][:, 0, 0]
            / buffer["episode_length"][:, 0, 0].sum(),
        )
        t_max_episodes = buffer["episode_length"][episode_idxs, 0].flatten()
        t_samples = np.random.randint(t_max_episodes)
        transitions = {
            key: buffer[key][episode_idxs, t_samples] for key in buffer.keys()
        }
        return transitions
