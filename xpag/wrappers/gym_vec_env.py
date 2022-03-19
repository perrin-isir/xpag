# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import sys
import inspect
import numpy as np
import gym
from gym.vector.utils import (
    write_to_shared_memory,
)
from gym.vector import VectorEnv, AsyncVectorEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
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


def gym_vec_env_(env_name, num_envs):
    if "num_envs" in inspect.signature(
        gym.envs.registration.load(
            gym.envs.registry.spec(env_name).entry_point
        ).__init__
    ).parameters and hasattr(
        gym.envs.registration.load(gym.envs.registry.spec(env_name).entry_point),
        "reset_done",
    ):
        # no need to create a VecEnv and wrap it if the env accepts 'num_envs' as an
        # argument at __init__ and has a reset_done() method.
        env = gym.make(env_name, num_envs=num_envs)
        # We force the environment to have a time limit, but
        # env.spec.max_episode_steps cannot exist as it would automatically trigger
        # the TimeLimit wrapper of gym, which does not handle batch envs. We require
        # max_episode_steps to be stored as an attribute of env:
        assert (
            (
                not hasattr(env.spec, "max_episode_steps")
                or env.spec.max_episode_steps is None
            )
            and hasattr(env, "max_episode_steps")
            and env.max_episode_steps is not None
        ), (
            "Trying to create a batch environment. env.max_episode_steps must exist, "
            "and env.spec.max_episode_steps must be None."
        )
        max_episode_steps = env.max_episode_steps
        env_type = "Gym"
    else:
        dummy_env = gym.make(env_name)
        # We force the env to have a standard gym time limit:
        assert (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ), "Only allowing gym envs with time limit (spec.max_episode_steps)."
        env = ResetDoneVecWrapper(
            AsyncVectorEnv(
                [lambda: ResetDoneWrapper(gym.make(env_name))] * num_envs,
                worker=_worker_shared_memory_no_auto_reset,
            )
        )
        env._spec = dummy_env.spec
        max_episode_steps = dummy_env.spec.max_episode_steps
        env_type = "Mujoco" if isinstance(dummy_env.unwrapped, MujocoEnv) else "Gym"
    is_goalenv = check_goalenv(env)
    env_info = {
        "env_type": env_type,
        "name": env_name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs,
        "max_episode_steps": max_episode_steps,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
    }
    get_env_dimensions(env_info, is_goalenv, env)
    return env, env_info


def gym_vec_env(env_name, num_envs):
    env, env_info = gym_vec_env_(env_name, num_envs)
    eval_env, _ = gym_vec_env_(env_name, 1)
    return env, eval_env, env_info


class ResetDoneVecWrapper(gym.Wrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def reset_done(self, **kwargs):
        return np.array(self.env.call("reset_done", **kwargs))

    def step(self, action):
        obs, reward, done, info_ = self.env.step(action)
        info = {
            "info_tuple": info_,
            "truncation": np.array(
                [
                    [elt["TimeLimit.truncated"] if "TimeLimit.truncated" in elt else 0]
                    for elt in info_
                ]
            ).reshape((self.env.num_envs, -1)),
        }

        return (
            obs.reshape((self.env.num_envs, -1)),
            reward.reshape((self.env.num_envs, -1)),
            done.reshape((self.env.num_envs, -1)),
            info,
        )


def _worker_shared_memory_no_auto_reset(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if "return_info" in data and data["return_info"] is True:
                    observation, info = env.reset(**data)
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    pipe.send(((None, info), True))
                else:
                    observation = env.reset(**data)
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    pipe.send((None, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # NO AUTOMATIC RESET
                # if done:
                #     info["terminal_observation"] = observation
                #     observation = env.reset()
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
