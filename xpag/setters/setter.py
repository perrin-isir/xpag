# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Tuple, Any
import os


class Setter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, env, observation, info=None, eval_mode=False) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def reset_done(
        self, env, observation, info=None, eval_mode=False
    ) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        done,
        info,
        eval_mode=False,
    ) -> Tuple[Any, Any, Any, Any]:
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


class DefaultSetter(Setter, ABC):
    def __init__(self):
        super().__init__("DefaultSetter")

    def reset(self, env, observation, info=None, eval_mode=False):
        if info is None:
            return observation
        else:
            return observation, info

    def reset_done(self, env, observation, info=None, eval_mode=False):
        if info is None:
            return observation
        else:
            return observation, info

    def step(
        self,
        env,
        observation,
        action,
        action_info,
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


class CompositeSetter(Setter, ABC):
    def __init__(self, setter1: Setter, setter2: Setter):
        super().__init__("CompositeSetter")
        self.setter1 = setter1
        self.setter2 = setter2

    def reset(self, env, observation, info=None, eval_mode=False):
        obs_, info_ = self.setter1.reset(env, observation, info, eval_mode)
        return self.setter2.reset(env, obs_, info_, eval_mode)

    def reset_done(self, env, observation, info=None, eval_mode=False):
        obs_, info_ = self.setter1.reset_done(env, observation, info, eval_mode)
        return self.setter2.reset_done(env, obs_, info_, eval_mode)

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        done,
        info,
        eval_mode=False,
    ):
        new_obs_, reward_, done_, info_ = self.setter1.step(
            env,
            observation,
            action,
            action_info,
            new_observation,
            reward,
            done,
            info,
            eval_mode,
        )
        return self.setter2.step(
            env,
            observation,
            action,
            action_info,
            new_obs_,
            reward_,
            done_,
            info_,
            eval_mode,
        )

    def write_config(self, output_file: str):
        self.setter1.write_config(output_file + ".1")
        self.setter2.write_config(output_file + ".2")

    def save(self, directory: str):
        self.setter1.save(os.path.join(directory, "1"))
        self.setter2.save(os.path.join(directory, "2"))

    def load(self, directory: str):
        self.setter1.load(os.path.join(directory, "1"))
        self.setter2.load(os.path.join(directory, "2"))
