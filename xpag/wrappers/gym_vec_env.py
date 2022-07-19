# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import sys
import inspect
import numpy as np
import gym
from gym.vector.utils import write_to_shared_memory, concatenate, create_empty_array
from gym.vector import VectorEnv, AsyncVectorEnv
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


def gym_vec_env_(env_name, num_envs, wrap_function=None):
    if wrap_function is None:

        def wrap_function(x):
            return x

    if "num_envs" in inspect.signature(
        gym.envs.registration.load(gym.spec(env_name).entry_point).__init__
    ).parameters and hasattr(
        gym.envs.registration.load(gym.spec(env_name).entry_point),
        "reset_done",
    ):
        # no need to create a VecEnv and wrap it if the env accepts 'num_envs' as an
        # argument at __init__ and has a reset_done() method.
        env = wrap_function(
            gym.make(env_name, num_envs=num_envs).unwrapped  # removing gym wrappers
        )

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
            "and env.spec.max_episode_steps must not (or be None)."
        )
        max_episode_steps = env.max_episode_steps
        env_type = "Gym"
    else:
        dummy_env = wrap_function(gym.make(env_name))
        # We force the env to have either a standard gym time limit (with the max number
        # of steps defined in .spec.max_episode_steps), or the max number of steps
        # defined in .max_episode_steps (and in this case we trust the environment
        # to appropriately prevent episodes from exceeding max_episode_steps steps).
        assert (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ) or (
            hasattr(dummy_env, "max_episode_steps")
            and dummy_env.max_episode_steps is not None
        ), (
            "Only allowing gym envs with time limit (defined in "
            ".spec.max_episode_steps or .max_episode_steps)."
        )
        env = wrap_function(
            ResetDoneVecWrapper(
                AsyncVectorEnv(
                    [
                        (lambda: gym.make(env_name))
                        if hasattr(dummy_env, "reset_done")
                        else (lambda: ResetDoneWrapper(gym.make(env_name)))
                    ]
                    * num_envs,
                    worker=_worker_shared_memory_no_auto_reset,
                )
            )
        )
        if (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ):
            max_episode_steps = dummy_env.spec.max_episode_steps
        else:
            max_episode_steps = dummy_env.max_episode_steps
        # env_type = "Mujoco" if isinstance(dummy_env.unwrapped, MujocoEnv) else "Gym"
        # To avoid imposing a dependency to mujoco, we simply guess that the
        # environment is a mujoco environment when it has the 'init_qpos', 'init_qvel'
        # and 'state_vector' attributes:
        env_type = (
            "Mujoco"
            if hasattr(dummy_env.unwrapped, "init_qpos")
            and hasattr(dummy_env.unwrapped, "init_qvel")
            and hasattr(dummy_env.unwrapped, "state_vector")
            else "Gym"
        )
        # The 'init_qpos' and 'state_vector' attributes are the one required to
        # save mujoco episodes (cf. class SaveEpisode in xpag/tools/eval.py).
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


def gym_vec_env(env_name, num_envs, wrap_function=None):
    env, env_info = gym_vec_env_(env_name, num_envs, wrap_function)
    eval_env, _ = gym_vec_env_(env_name, 1, wrap_function)
    return env, eval_env, env_info


class ResetDoneVecWrapper(gym.Wrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)

    def reset(self, **kwargs):
        if "return_info" in kwargs and kwargs["return_info"]:
            obs, info_ = self.env.reset(**kwargs)
            return obs, {"info_tuple": tuple(info_)}
        else:
            return self.env.reset(**kwargs)

    def reset_done(self, *args, **kwargs):
        if "return_info" in kwargs and kwargs["return_info"]:
            results, info_ = tuple(zip(*self.env.call("reset_done", *args, **kwargs)))
        else:
            results = self.env.call("reset_done", *args, **kwargs)
        observations = create_empty_array(
            self.env.single_observation_space, n=self.num_envs, fn=np.empty
        )
        if "return_info" in kwargs and kwargs["return_info"]:
            info = {"info_tuple": tuple(info_)}
            return (
                concatenate(self.env.single_observation_space, results, observations),
                info,
            )
        else:
            return concatenate(self.env.single_observation_space, results, observations)

    def step(self, action):
        obs, reward, done, info_ = self.env.step(action)
        info_["truncation"] = (
            info_["TimeLimit.truncated"]
            if "TimeLimit.truncated" in info_
            else np.array([False] * self.num_envs).reshape((self.num_envs, 1))
        )
        info_["is_success"] = (
            info_["is_success"]
            if "is_success" in info_
            else np.array([False] * self.num_envs).reshape((self.num_envs, 1))
        )

        return (
            obs,
            reward.reshape((self.env.num_envs, -1)),
            done.reshape((self.env.num_envs, -1)),
            info_,
        )


def _worker_shared_memory_no_auto_reset(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    """
    This function is derived from _worker_shared_memory() in gym. See:
    https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py
    """
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
            # elif command == "seed":
            #     env.seed(data)
            #     pipe.send((None, True))
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
                function = getattr(env, name)
                if name == "reset_done":
                    pipe.send((function(index, *args, **kwargs), True))
                else:
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
                    "be one of {`reset`, `step`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
