# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

import sys
import inspect
import multiprocessing
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium.vector.utils import (
    write_to_shared_memory,
    concatenate,
    create_empty_array,
)
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
import traceback
from gymnasium.vector import VectorEnv, AsyncVectorEnv, VectorWrapper
from xpag.wrappers.reset_done import ResetDoneWrapper
from xpag.tools.utils import get_env_dimensions


def check_goalenv(env) -> bool:
    """
    Checks if an environment is of type 'GoalEnv'.
    The migration of GoalEnv from gym (0.22) to gym-robotics makes this verification
    non-trivial. Here we just verify that the observation_space has a structure
    that is compatible with the GoalEnv class.
    """
    if isinstance(env, VectorEnv):
        obs_space = env.single_observation_space
    else:
        obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Dict):
        return False
    else:
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in obs_space.spaces:
                return False
    return True


def gym_vec_env_(env_name, 
                 num_envs, 
                 wrap_function=None, 
                 max_episode_steps=None,
                 **gym_kwargs):
    if wrap_function is None:

        def wrap_function(x):
            return x

    env_creator = gym.spec(env_name).entry_point
    if type(env_creator) == str:
        env_creator = gym.envs.registration.load_env_creator(env_creator)

    if "num_envs" in inspect.signature(
        env_creator.__init__
    ).parameters and hasattr(
        env_creator,
        "reset_done",
    ):
        # no need to create a VecEnv and wrap it if the env accepts 'num_envs' as an
        # argument at __init__ and has a reset_done() method. In this case, we trust
        # the environment to properly handle parallel rollouts.

        env = wrap_function(
            gym.make(
                env_name, num_envs=num_envs, **gym_kwargs
            ).unwrapped  # removing gymnasium wrappers
        )

        # We force the environment to have a time limit, but
        # env.spec.max_episode_steps cannot exist as it would automatically trigger
        # the TimeLimit wrapper of gymnasium, which does not handle batch envs.
        # We require max_episode_steps to be stored as an attribute of env:
        assert (
            (
                not hasattr(env.spec, "max_episode_steps")
                or env.spec.max_episode_steps is None
            )
            and hasattr(env, "max_episode_steps")
            and env.max_episode_steps is not None
        ), (
            "Trying to create a batch environment. env.max_episode_steps must exist, "
            "and env.spec.max_episode_steps must not (or be None)."
        )
        env_type = "Gym"
    else:
        dummy_env = gym.make(env_name, **gym_kwargs)
        # We force the env to either have a standard gymnasium time limit (with the
        # max number of steps defined in .spec.max_episode_steps), or the max number of
        # steps defined in .max_episode_steps (and in this case we trust the environment
        # to appropriately prevent episodes from exceeding max_episode_steps steps). 
        # Otherwise, the argument max_episode_steps must not be None.
        if (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ):
            max_episode_steps = dummy_env.spec.max_episode_steps
        elif (
            hasattr(dummy_env, "max_episode_steps") 
            and dummy_env.max_episode_steps is not None
        ):
            max_episode_steps = dummy_env.max_episode_steps
        else:
            assert (
                max_episode_steps is not None
            ), (
                "Error: episode step limit undefined, so argument "
                "max_episode_steps cannot be None."
            )
        # env_type = "Mujoco" if isinstance(dummy_env.unwrapped, MujocoEnv) else "Gym"
        # To avoid imposing a dependency to mujoco, we simply guess that the
        # environment is a mujoco environment when it has the 'init_qpos', 'init_qvel',
        # 'state_vector', 'do_simulation' and 'get_body_com' attributes:
        env_type = (
            "Mujoco"
            if hasattr(dummy_env.unwrapped, "init_qpos")
            and hasattr(dummy_env.unwrapped, "init_qvel")
            and hasattr(dummy_env.unwrapped, "state_vector")
            and hasattr(dummy_env.unwrapped, "do_simulation")
            and hasattr(dummy_env.unwrapped, "get_body_com")
            else "Gym"
        )
        # The 'init_qpos' and 'state_vector' attributes are the one required to
        # save mujoco episodes (cf. class SaveEpisode in xpag/tools/eval.py).
        env = wrap_function(
            ResetDoneVecWrapper(
                AsyncVectorEnv(
                    [
                        (lambda: gym.make(env_name, **gym_kwargs))
                        if hasattr(dummy_env, "reset_done")
                        else (
                            lambda: ResetDoneWrapper(gym.make(env_name, **gym_kwargs))
                        )
                    ]
                    * num_envs,
                    worker=_worker_shared_memory_no_auto_reset,
                    # observation_mode="same"
                ),
                max_episode_steps,
            )
        )

    is_goalenv = check_goalenv(env)
    env_info = {
        "env_type": env_type,
        "name": env_name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs,
        "max_episode_steps": env.max_episode_steps,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
    }
    get_env_dimensions(env_info, is_goalenv, env)
    return env, env_info


def gym_vec_env(env_name: str, num_envs: int, wrap_function: Callable = None, **kwargs):
    env, env_info = gym_vec_env_(env_name, num_envs, wrap_function, **kwargs)
    eval_env, _ = gym_vec_env_(env_name, 1, wrap_function, **kwargs)
    return env, eval_env, env_info


class ResetDoneVecWrapper(VectorWrapper):
    def __init__(self, env: VectorEnv, max_episode_steps: int):
        super().__init__(env)
        self.steps = np.zeros(self.num_envs).reshape((self.num_envs,-1))
        self.max_episode_steps = max_episode_steps

    def reset(self, **kwargs):
        obs, info_ = self.env.reset(**kwargs)
        self.steps = np.zeros(self.num_envs).reshape((self.num_envs,-1))
        return obs, info_

    def reset_done(self, *args, **kwargs):
        done = args[0]
        self.steps = self.steps * (1 - done)
        results, info_ = tuple(zip(*self.env.call("reset_done", *args, **kwargs)))
        observations = create_empty_array(
            self.env.single_observation_space, n=self.num_envs, fn=np.empty
        )
        infos = {}
        for i, d in enumerate(info_):
            self._add_info(infos, d, i)
        return (
            concatenate(self.env.single_observation_space, results, observations),
            infos,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info_ = self.env.step(action)
        info_["is_success"] = (
            info_["is_success"].flatten()
            if "is_success" in info_
            else np.array([False] * self.num_envs).flatten()
        )
        self.steps += 1
        truncated = np.logical_or(
            truncated.reshape((self.env.num_envs, -1)), 
            self.steps >= self.max_episode_steps)
        return (
            obs,
            reward.reshape((self.env.num_envs, -1)),
            terminated.reshape((self.env.num_envs, -1)),
            truncated,
            info_,
        )

def _worker_shared_memory_no_auto_reset(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue, autoreset_mode=None
):
    """
    This function is derived from _async_worker() in gymnasium. See:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/async_vector_env.py
    """
    assert(shared_memory is not None)

    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    observation = None

    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # NO AUTOMATIC RESET
                # if terminated or truncated:
                #     old_observation = observation
                #     observation, info = env.reset()
                #     info["final_observation"] = old_observation
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = env.get_wrapper_attr(name)
                # function = getattr(env, name)
                if name == "reset_done":
                    pipe.send((function(index, *args, **kwargs), True))
                else:
                    if callable(function):
                        pipe.send((function(*args, **kwargs), True))
                    else:
                        pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                env.set_wrapper_attr(name, value)
                # setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                obs_mode, single_obs_space, single_action_space = data

                pipe.send(
                    (
                        (
                            (
                                single_obs_space == observation_space
                                if obs_mode == "same"
                                else is_space_dtype_shape_equiv(
                                    single_obs_space, observation_space
                                )
                            ),
                            single_action_space == action_space,
                        ),
                        True,
                    )
                )
                # pipe.send(
                #     ((data[0] == observation_space, data[1] == env.action_space), True)
                # )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()

        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()
