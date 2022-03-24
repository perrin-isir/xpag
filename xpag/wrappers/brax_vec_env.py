# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from typing import ClassVar, Optional
from xpag.wrappers.reset_done import ResetDoneBraxWrapper
from brax import jumpy as jp
import jax
import gym
from gym import spaces
from gym.vector import utils
from brax import envs
from xpag.tools.utils import get_env_dimensions

_envs_episode_length = {
    "acrobot": 1000,
    "ant": 1000,
    "fast": 1000,
    "fetch": 1000,
    "grasp": 1000,
    "halfcheetah": 1000,
    "hopper": 1000,
    "humanoid": 1000,
    "humanoidstandup": 1000,
    "inverted_pendulum": 1000,
    "inverted_double_pendulum": 1000,
    "reacher": 1000,
    "reacherangle": 1000,
    "swimmer": 1000,
    "ur5e": 1000,
    "walker2d": 1000,
}


def brax_vec_env_(env_name, num_envs, force_cpu_backend=False):
    assert env_name in _envs_episode_length, f"{env_name}: unknown environment."
    env = ResetDoneBraxToGymWrapper(
        ResetDoneBraxWrapper(
            envs.create(
                env_name=env_name,
                episode_length=_envs_episode_length[env_name],
                batch_size=num_envs,
                auto_reset=False,
            )
        ),
        backend="cpu" if force_cpu_backend else None,
    )
    is_goalenv = False  # No Brax GoalEnv support so far
    env_info = {
        "env_type": "Brax",
        "name": env_name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs,
        "max_episode_steps": _envs_episode_length[env_name],
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
    }
    get_env_dimensions(env_info, is_goalenv, env)
    return env, env_info


def brax_vec_env(env_name, num_envs, force_cpu_backend=False):
    env, env_info = brax_vec_env_(env_name, num_envs, force_cpu_backend)
    eval_env, _ = brax_vec_env_(env_name, 1, force_cpu_backend)
    return env, eval_env, env_info


class ResetDoneBraxToGymWrapper(gym.Env):
    """
    A wrapper that converts Brax Env to one that follows Gym VectorEnv API,
    with the additional reset_done() method.
    """

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: ResetDoneBraxWrapper, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.sys.config.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size
        self.backend = backend
        self._state = None
        self._key = None

        obs_high = jp.inf * jp.ones(self._env.observation_size, dtype="float32")
        self.single_observation_space = spaces.Box(-obs_high, obs_high, dtype="float32")
        self.observation_space = utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        action_high = jp.ones(self._env.action_size, dtype="float32")
        self.single_action_space = spaces.Box(
            -action_high, action_high, dtype="float32"
        )
        self.action_space = utils.batch_space(self.single_action_space, self.num_envs)

        def reset(key):
            key1, key2 = jp.random_split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = state.metrics
            info["steps"] = state.info["steps"]
            info["truncation"] = state.info["truncation"]
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

        def reset_done(state, key):
            key1, key2 = jp.random_split(key)
            if state is None:
                raise ValueError("Use reset() for the first reset, not reset_done().")
            state = self._env.reset_done(state, key2)
            return state, state.obs, key1

        self._reset_done = jax.jit(reset_done, backend=self.backend)

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            if self._key is None:
                self._key = jax.random.PRNGKey(0)
        else:
            self._key = jax.random.PRNGKey(seed)
        self._state, obs, self._key = self._reset(self._key)
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        info["truncation"] = info["truncation"].reshape((self.num_envs, -1))
        return (
            obs.reshape((self.num_envs, -1)),
            reward.reshape((self.num_envs, -1)),
            done.reshape((self.num_envs, -1)),
            info,
        )

    def reset_done(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            if self._key is None:
                self._key = jax.random.PRNGKey(0)
        else:
            self._key = jax.random.PRNGKey(seed)
        self._state, obs, self._key = self._reset_done(self._state, self._key)
        return obs

    def render(self, mode="human"):
        # pylint:disable=g-import-not-at-top
        from brax.io import image

        if mode == "rgb_array":
            sys = self._env.sys
            qp = jp.take(self._state.qp, 0)
            return image.render_array(sys, qp, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception
