# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from xpag.tools.utils import DataType


class GoalSetter(ABC):
    def __init__(self, name: str, params: dict,
                 num_envs: int = 1,
                 datatype: DataType = DataType.TORCH,
                 device: str = 'cpu'):
        self.name = name
        self.params = params
        self.num_envs = num_envs
        self.datatype = datatype
        self.device = device

    @abstractmethod
    def reset(self, obs):
        pass

    @abstractmethod
    def step(self, o, action, new_o, reward, done, info):
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


class DefaultGoalSetter(GoalSetter, ABC):
    def __init__(self, params=None,
                 num_envs: int = 1,
                 datatype: DataType = DataType.TORCH,
                 device: str = 'cpu'):
        if params is None:
            params = {}
        super().__init__("DefaultGoalSetter", params, num_envs, datatype, device)

    def reset(self, obs):
        return obs

    def step(self, o, action, new_o, reward, done, info):
        return o, action, new_o, reward, done, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass
