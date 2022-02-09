import numpy as np
import torch
import gym
import gym_gmazes
import functools
from brax import envs
from brax.envs import to_torch
from xpag.tools.utils import DataType
from xpag.tools.learn import check_goalenv, get_dimensions, default_replay_buffer
from xpag.samplers import DefaultSampler, HER
from xpag.agents import SAC
import re


def configure(
        env_name_, num_envs_, gmaze_frame_skip_, gmaze_walls_,
        episode_max_length_, buffer_name_, buffer_size_,
        sampler_name_, goalenv_sampler_name_, agent_name_,
        seed_=None
):
    if env_name_.startswith('brax-'):
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        # torch allocation on device first, to prevent JAX from swallowing up all the
        # GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
        # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        v_ = torch.ones(1, device=device_)
        # print(torch.cuda.memory_allocated(device='cuda'), 'bytes')
        env_true_name = re.sub('-v.$', '', env_name_).removeprefix('brax-')
        gym_name = env_name_
        if gym_name not in gym.envs.registry.env_specs:
            entry_point = functools.partial(envs.create_gym_env, env_name=env_true_name)
            gym.register(gym_name, entry_point=entry_point)
        env_ = gym.make(gym_name, batch_size=num_envs_,
                        episode_length=episode_max_length_)
        # automatically convert between jax ndarrays and torch tensors:
        env_ = to_torch.JaxToTorchWrapper(env_, device=device_)
        datatype_ = DataType.TORCH
    elif env_name_.startswith('GMaze'):
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_ = gym.make("GMazeSimple-v0",
                        device=device_,
                        batch_size=num_envs_,
                        frame_skip=gmaze_frame_skip_,
                        walls=gmaze_walls_)
        datatype_ = DataType.TORCH
    else:
        if num_envs_ > 1:
            env_ = gym.vector.make(env_name_, num_envs=num_envs_)
            env_.spec = gym.envs.registration.EnvSpec(env_name_)
        else:
            env_ = gym.make(env_name_)
        datatype_ = DataType.NUMPY
        device_ = 'cpu'

    agent_params = {}
    # Set seeds
    if seed_ is not None:
        env_.seed(seed_)
        torch.manual_seed(seed_)
        np.random.seed(seed_)
        agent_params['seed'] = seed_

    is_goalenv = check_goalenv(env_)
    dimensions = get_dimensions(env_)

    if buffer_name_ == 'DefaultBuffer':
        replay_buffer_ = default_replay_buffer(
            buffer_size_,
            episode_max_length_,
            env_,
            datatype_,
            device_
        )
    else:
        replay_buffer_ = None  # only one available buffer so far

    if is_goalenv:
        sampler_ = eval(goalenv_sampler_name_)(
            env_.compute_reward, datatype=datatype_)
    else:
        sampler_ = eval(sampler_name_)(datatype=datatype_)

    if is_goalenv:
        agent_ = eval(agent_name_)(
            dimensions['observation_dim'] + dimensions['action_dim'],
            dimensions['action_dim'],
            params=agent_params)
    else:
        agent_ = eval(agent_name_)(dimensions['observation_dim'],
                                   dimensions['action_dim'],
                                   params=agent_params)

    return agent_, env_, replay_buffer_, sampler_, datatype_, device_
