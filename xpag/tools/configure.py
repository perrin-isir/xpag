# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
import torch
import jax
import gym
import functools
from brax import envs
from brax.envs import to_torch
from xpag.tools.utils import DataType
from xpag.tools.learn import check_goalenv, get_dimensions, default_replay_buffer
import re


def configure(
    env_name_,
    num_envs_,
    episode_max_length_,
    buffer_size_,
    sampler_class_,
    agent_class_,
    goalsetter_class_,
    device_: str = "cpu",
    seed_=None,
):
    if seed_ is not None:
        torch.manual_seed(seed_)
        np.random.seed(seed_)

    continue_after_done_ = False

    if env_name_.startswith("brax-"):
        # brax environment
        # torch allocation on device first, to prevent JAX from swallowing up all the
        # GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
        # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        v_ = torch.ones(1, device=device_)
        assert v_
        # print(torch.cuda.memory_allocated(device='cuda'), 'bytes')
        env_true_name = re.sub("-v.$", "", env_name_).removeprefix("brax-")
        gym_name = env_name_
        if gym_name not in gym.envs.registry.env_specs:
            entry_point = functools.partial(
                envs.create_gym_env,
                env_name=env_true_name,
                backend="gpu"
                if (
                    device_ == "cuda"
                    and jax.lib.xla_bridge.get_backend().platform == "gpu"
                )
                else "cpu",
            )
            gym.register(gym_name, entry_point=entry_point)
        env_ = gym.make(
            gym_name, batch_size=num_envs_, episode_length=episode_max_length_
        )
        # automatically convert between jax ndarrays and torch tensors:
        env_ = to_torch.JaxToTorchWrapper(env_, device=device_)
        datatype_ = DataType.TORCH
    elif env_name_.startswith("GMaze"):
        # GMaze environment
        env_ = gym.make(env_name_, device=device_, batch_size=num_envs_)
        datatype_ = DataType.TORCH
        continue_after_done_ = True
    else:
        # mujoco environment
        env_ = gym.vector.make(env_name_, num_envs=num_envs_)
        env_.spec = gym.envs.registration.EnvSpec(env_name_)
        datatype_ = DataType.NUMPY

    backend = jax.lib.xla_bridge.get_backend().platform
    agent_params = {"backend": backend}
    goalsetter_params = {}
    # Set seeds
    if seed_ is not None:
        env_.seed(seed_)
        env_.action_space.seed(seed_)
        env_.observation_space.seed(seed_)
        agent_params["seed"] = seed_
        goalsetter_params["seed"] = seed_

    is_goalenv = check_goalenv(env_)
    dimensions = get_dimensions(env_)

    replay_buffer_ = default_replay_buffer(
        buffer_size_, episode_max_length_, env_, datatype_, device_
    )

    if is_goalenv:
        sampler_ = sampler_class_(env_.compute_reward, datatype=datatype_)
    else:
        sampler_ = sampler_class_(datatype=datatype_)

    if is_goalenv:
        agent_ = agent_class_(
            dimensions["observation_dim"] + dimensions["desired_goal_dim"],
            dimensions["action_dim"],
            params=agent_params,
        )
    else:
        agent_ = agent_class_(
            dimensions["observation_dim"], dimensions["action_dim"], params=agent_params
        )

    goalsetter_params["agent"] = agent_

    goalsetter_ = goalsetter_class_(
        params=goalsetter_params, num_envs=num_envs_, datatype=datatype_, device=device_
    )

    return (
        agent_,
        goalsetter_,
        env_,
        continue_after_done_,
        replay_buffer_,
        sampler_,
        datatype_,
    )
