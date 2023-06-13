# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

from typing import ClassVar, Optional, Union, List, Callable
from xpag.wrappers.gym_vec_env import check_goalenv
import jax
import jax.numpy as jnp
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import utils
from xpag.tools.utils import get_env_dimensions
from brax import envs
from brax.envs import Wrapper, State

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


def brax_vec_env_(
    env_name: str,
    num_envs: int,
    wrap_function: Callable = None,
    *,
    force_cpu_backend=False,
):
    class ResetDoneBraxWrapper(Wrapper):
        """Adds reset_done() to Brax envs."""

        def reset(self, rng: jnp.ndarray) -> State:
            state = self.env.reset(rng)
            return state

        def step(self, state: State, action: jnp.ndarray) -> State:
            return self.env.step(state, action)

        def reset_done(self, done: jnp.ndarray, state: State, rng: jnp.ndarray):
            # done = state.done
            def where_done(x, y):
                done_ = done
                if done_.shape:
                    done_ = jnp.reshape(
                        done_, tuple([x.shape[0]] + [1] * (len(x.shape) - 1))
                    )  # type: ignore
                return jnp.where(done_, x, y)

            if "steps" in state.info:
                steps = state.info["steps"]
                steps = where_done(jnp.zeros_like(steps), steps)
                state.info.update(steps=steps)

            reset_state = self.env.reset(rng)
            new_state = jax.tree_util.tree_map(where_done, reset_state, state)
            return new_state

    class ResetDoneBraxToGymWrapper(gym.vector.VectorEnv):
        """
        A wrapper that converts Brax Env to one that follows Gym VectorEnv API,
        with the additional reset_done() and reset_idxs() methods.
        """

        # Flag that prevents `gym.register` from misinterpreting the `_step` and
        # `_reset` as signs of a deprecated gym Env API.
        _gym_disable_underscore_compat: ClassVar[bool] = True

        def __init__(
            self,
            env: ResetDoneBraxWrapper,
            max_episode_steps: int,
            backend: Optional[str] = None,
        ):
            self.max_episode_steps = max_episode_steps
            self._env = env
            if not hasattr(self._env, "batch_size"):
                raise ValueError("underlying env must be batched")

            self.num_envs = self._env.batch_size
            self.backend = backend
            self._state = None
            self._key = None

            obs_high = np.inf * np.ones(self._env.observation_size, dtype="float32")
            self.single_observation_space = spaces.Box(
                -obs_high, obs_high, dtype=np.float32
            )
            self.observation_space = utils.batch_space(
                self.single_observation_space, self.num_envs
            )

            action_high = np.ones(self._env.action_size, dtype="float32")
            self.single_action_space = spaces.Box(
                -action_high, action_high, dtype=np.float32
            )
            self.action_space = utils.batch_space(
                self.single_action_space, self.num_envs
            )

            def reset(key):
                key1, key2 = jax.random.split(key)
                state = self._env.reset(key2)
                return state, state.obs, key1

            self._reset = jax.jit(reset, backend=self.backend)

            def step(state, action):
                state = self._env.step(state, action)
                info = state.metrics.copy()
                info["steps"] = state.info["steps"]
                terminated = jnp.logical_and(
                    state.done, jnp.logical_not(state.info["truncation"])
                ).reshape((self.num_envs, -1))
                # terminated has the wrong value here if the episode was both truncated
                # and reached a terminal state. However, with the current API of brax
                # envs, this information cannot be recovered.
                truncated = (
                    state.info["truncation"].reshape((self.num_envs, -1)).astype("bool")
                )
                return (
                    state,
                    state.obs.reshape((self.num_envs, -1)),
                    state.reward.reshape((self.num_envs, -1)),
                    terminated,
                    truncated,
                    info,
                )

            self._step = jax.jit(step, backend=self.backend)

            def reset_done(done, state, key):
                key1, key2 = jax.random.split(key)
                if state is None:
                    raise ValueError(
                        "Use reset() for the first reset, not reset_idxs()."
                    )
                state = self._env.reset_done(done, state, key2)
                return state, state.obs, key1

            self._reset_done = jax.jit(reset_done, backend=self.backend)

        def reset(
            self,
            *,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
        ):
            if seed is None:
                if self._key is None:
                    self._key = jax.random.PRNGKey(0)
            else:
                if isinstance(seed, int):
                    self._key = jax.random.PRNGKey(seed)
                else:
                    self._key = jax.random.PRNGKey(
                        seed[0]  # only a single seed is needed
                    )
            self._state, obs, self._key = self._reset(self._key)
            return obs, {}

        def step(self, action):
            self._state, obs, reward, terminated, truncated, info = self._step(
                self._state, action
            )
            return (
                obs,
                reward,
                terminated,
                truncated,
                info,
            )

        def reset_done(
            self,
            done,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
            if seed is None:
                if self._key is None:
                    self._key = jax.random.PRNGKey(0)
            else:
                self._key = jax.random.PRNGKey(seed)
            self._state, obs, self._key = self._reset_done(done, self._state, self._key)
            return obs, {}

    if wrap_function is None:

        def wrap_function(x):
            return x

    assert env_name in _envs_episode_length, f"{env_name}: unknown environment."
    base_env = envs.create(
        env_name=env_name,
        episode_length=_envs_episode_length[env_name],
        batch_size=num_envs,
        auto_reset=False,
    )
    env = wrap_function(
        ResetDoneBraxToGymWrapper(
            ResetDoneBraxWrapper(base_env),
            _envs_episode_length[env_name],
            backend="cpu" if force_cpu_backend else None,
        )
    )
    is_goalenv = check_goalenv(env)
    env_info = {
        "env_type": "Brax",
        "name": env_name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs,
        "max_episode_steps": env.max_episode_steps,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
    }
    get_env_dimensions(env_info, is_goalenv, env)
    return env, env_info


def brax_vec_env(env_name, num_envs, wrap_function=None, force_cpu_backend=False):
    env, env_info = brax_vec_env_(
        env_name, num_envs, wrap_function, force_cpu_backend=force_cpu_backend
    )
    eval_env, _ = brax_vec_env_(
        env_name, 1, wrap_function, force_cpu_backend=force_cpu_backend
    )
    return env, eval_env, env_info
