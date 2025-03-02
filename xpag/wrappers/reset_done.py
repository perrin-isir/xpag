# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import gymnasium as gym
from typing import Union


class ResetDoneWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int | None=None):
        super().__init__(env)
        self._last_obs = None
        self.steps = 0
        self.max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.steps = 0
        return obs, info

    def reset_done(self, index, done: np.ndarray, **kwargs):
        if done[index]:
            obs, info = self.env.reset(**kwargs)
            self._last_obs = obs
            self.steps = 0
        else:
            info = {}
        return self._last_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self.steps += 1
        info["steps"] = self.steps
        if self.max_episode_steps and self.steps >= self.max_episode_steps:
            truncated = True 
        return obs, reward, terminated, truncated, info
