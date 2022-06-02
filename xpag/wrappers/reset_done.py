# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
from brax import jumpy as jp
from brax.envs import env as brax_env
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


class ResetDoneBraxWrapper(brax_env.Wrapper):
    """Adds reset_done() to Brax envs."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info["first_qp"] = state.qp
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        return self.env.step(state, action)

    def reset_done(self, done: jp.ndarray, state: brax_env.State, rng: jp.ndarray):
        # done = state.done
        def where_done(x, y):
            done_ = done
            if done_.shape:
                done_ = jp.reshape(
                    done_, [x.shape[0]] + [1] * (len(x.shape) - 1)
                )  # type: ignore
            return jp.where(done_, x, y)

        if "steps" in state.info:
            steps = state.info["steps"]
            steps = where_done(jp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        reset_state = self.env.reset(rng)
        qp = jp.tree_map(where_done, reset_state.qp, state.qp)
        obs = where_done(reset_state.obs, state.obs)
        state = state.replace(qp=qp, obs=obs)
        return state.replace(done=where_done(jp.zeros_like(state.done), state.done))
