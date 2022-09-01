# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import gym


class ResetDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._last_done = True
        self._last_obs = None
        self.steps = 0

    def reset(self, **kwargs):
        if "return_info" in kwargs and kwargs["return_info"]:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        self._last_done = False
        self._last_obs = obs
        self.steps = 0
        if "return_info" in kwargs and kwargs["return_info"]:
            return obs, info
        else:
            return obs

    def reset_done(self, index, done: np.ndarray, **kwargs):
        # if self._last_done:
        if done[index]:
            if "return_info" in kwargs and kwargs["return_info"]:
                obs, info = self.env.reset(**kwargs)
            else:
                obs = self.env.reset(**kwargs)
            self._last_done = False
            self._last_obs = obs
            self.steps = 0
        else:
            info = {}
        if "return_info" in kwargs and kwargs["return_info"]:
            return self._last_obs, info
        else:
            return self._last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self._last_done = True
        self._last_obs = obs
        self.steps += 1
        info["steps"] = self.steps
        return obs, reward, done, info
