# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Tuple


class GoalSetter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, env, observation, eval_mode=False):
        pass

    @abstractmethod
    def reset_done(self, env, observation, eval_mode=False):
        pass

    @abstractmethod
    def step(
        self,
        env,
        observation,
        action,
        new_observation,
        reward,
        done,
        info,
        eval_mode=False,
    ) -> Tuple:
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
    def __init__(self):
        super().__init__("DefaultGoalSetter")

    def reset(self, env, observation, eval_mode=False):
        return observation

    def reset_done(self, env, observation, eval_mode=False):
        return observation

    def step(
        self,
        env,
        observation,
        action,
        new_observation,
        reward,
        done,
        info,
        eval_mode=False,
    ):
        return new_observation, reward, done, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass
