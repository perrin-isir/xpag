from datetime import datetime
from IPython import embed
import numpy as np
import torch
import gym
import gym_gmazes
import string
import collections
import functools
import random
import time
from datetime import datetime
import brax
import jax
from brax import envs
from brax.envs import to_torch
from brax.io import metrics
# from brax.io import torch
from IPython import embed
import argparse
import os
import logging
import xpag
from xpag.plotting.basics import plot_episode_2d
import re

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
print(torch.cuda.memory_allocated(device='cuda'), 'bytes')

#
# def str2bool(val):
#     if isinstance(val, bool):
#         return val
#     if val.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif val.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("Boolean value expected.")
#
#
# def get_args(rnddir):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tag", default="t01")
#     parser.add_argument(
#         "--env_name",
#         default="InvertedPendulum-v2"
#         # default="SimpleMazeGoalEnv-v0"
#         # default="HalfCheetahBulletEnv-v2",
#     )  # OpenAI gym environment name
#     parser.add_argument("--agent_name", default="TD3")  # Policy name
#     parser.add_argument("--device", default=None)  # "cuda" or "cpu"
#     parser.add_argument("--sampler_name", default="DefaultSampler")
#     parser.add_argument("--buffer_name", default="DefaultBuffer")
#     parser.add_argument("--goalsetter_name", default="DefaultGoalSetter")
#     parser.add_argument("--seed", type=int)  # Sets Gym, PyTorch and Numpy seeds
#     parser.add_argument(
#         "--load", type=str
#     )  # Load agent, goal setter and buffer with a given ID
#
#     parser.add_argument(
#         "--render", type=str2bool, nargs="?", const=True, default=False
#     )  # Activate rendering
#     parser.add_argument(
#         "--realtime", type=str2bool, nargs="?", const=True, default=False
#     )  # If True, episodes are rendered at 60 hz
#     parser.add_argument(
#         "--plotpaths", type=str2bool, nargs="?", const=True, default=False
#     )  # If True, during training every episode is plotted and saved in a .png file
#     parser.add_argument(
#         "--start_timesteps", default=10_000, type=int
#     )  # How many time steps purely random policy is run for
#     parser.add_argument(
#         "--eval_freq", default=5000, type=float
#     )  # How often (time steps) we evaluate
#     parser.add_argument(
#         "--save_freq", default=np.inf, type=float
#     )  # How often (time steps) models and data (agent + goal setter + buffer) are saved
#     parser.add_argument(
#         "--max_timesteps", default=1e6, type=float
#     )  # Max time steps to run environment for
#     parser.add_argument(
#         "--buffer_size", default=1e6, type=float
#     )  # Replay buffer size
#     parser.add_argument(
#         "--batch_size", default=256, type=int
#     )  # Batch size for both actor and critic
#     parser.add_argument(
#         "--train_ratio", default=1.0, type=float
#     )  # Number of training batches per step
#     parser.add_argument(
#         "--output_dir",
#         default=os.path.join(os.path.expanduser("~"), "results", "SGE_experiments"),
#     )
#     parser.add_argument(
#         "--save_dir",
#         default="save_dir",
#     )
#     args = parser.parse_args()
#     args.save_dir = os.path.expanduser(
#         os.path.join(args.output_dir, args.save_dir, args.env_name, args.tag, rnddir)
#     )
#     return args
#
#
# args = get_args('')

# env_name = 'halfcheetah'
# device = 'cuda'
# episode_max_length = 1000
# num_envs = 64
# gym_name = f'brax-{env_name}-v0'
# if gym_name not in gym.envs.registry.env_specs:
#     entry_point = functools.partial(envs.create_gym_env, env_name=env_name)
#     gym.register(gym_name, entry_point=entry_point)
# env = gym.make(gym_name, batch_size=num_envs, episode_length=episode_max_length)
# # automatically convert between jax ndarrays and torch tensors:
# env = to_torch.JaxToTorchWrapper(env, device=device)
# version = 'torch'
# datatype = xpag.tl.DataType.TORCH

# device = 'cuda'
# num_envs = 32
# episode_max_length = 50
# env = gym.make("GMazeSimple-v0",
#                device=device,
#                batch_size=num_envs,
#                frame_skip=2,
#                walls=[])
# datatype = xpag.tl.DataType.TORCH

# device = 'cuda'
# num_envs = 1024
# episode_max_length = 50
# env = gym.make("GMazeGoalSimple-v0",
#                device=device,
#                batch_size=num_envs,
#                frame_skip=2,
#                walls=None)
# datatype = xpag.tl.DataType.TORCH
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# episode_max_length = 1000
# num_envs = 1
# # gym.vector.make("CartPole-v1", num_envs=3)
# env = gym.make('HalfCheetah-v3')
# # env = gym.vector.make('HalfCheetah-v3', num_envs=num_envs)
# version = 'numpy'
# datatype = xpag.tl.DataType.NUMPY
#
# agent_params = {}
# # Set seeds
# args.seed = 0
# if args.seed is not None:
#     env.seed(args.seed)
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     agent_params['seed'] = args.seed
#
# is_goalenv = xpag.tl.check_goalenv(env)
#
# dimensions = xpag.tl.get_dimensions(env)
#
# replay_buffer = xpag.tl.default_replay_buffer(
#     args.buffer_size,
#     episode_max_length,
#     env,
#     datatype,
#     device
# )
#
# if is_goalenv:
#     sampler = xpag.sa.HER(env.compute_reward, datatype=datatype)
# else:
#     sampler = xpag.sa.DefaultSampler(datatype=datatype)
#
# if is_goalenv:
#     agent = xpag.ag.SAC(
#         dimensions['observation_dim'] + dimensions['action_dim'],
#         dimensions['action_dim'],
#         params=agent_params)
# else:
#     # agent = xpag.ag.SAC(dimensions['observation_dim'],
#     #                     dimensions['action_dim'], device,
#     #                     params=None)
#     # agent = xpag.ag.SAC_jax(dimensions['observation_dim'],
#     #                        dimensions['action_dim'], device,
#     #                        params=None)
#     agent = xpag.ag.SAC(dimensions['observation_dim'],
#                         dimensions['action_dim'],
#                         params=agent_params)

# save_dir = os.path.join(os.path.expanduser("~"),
#                         "results",
#                         "xpag",
#                         datetime.now().strftime("%Y%m%d_%H%M%S"))
#
# plot_episode = functools.partial(
#     plot_episode_2d,
#     plot_env_function=env.plot if hasattr(env, "plot") else None
# )
# plot_episode = None
# max_t = int(1e6)
# train_ratio = 1.
# batch_size = 256
# start_random_t = 0
# # eval_freq = 50 * 7
# eval_freq = 1000 * 5
# eval_eps = 5
# save_freq = 0
#
# # embed()
#
# xpag.tl.learn(agent, env, num_envs, episode_max_length,
#               max_t, train_ratio, batch_size, start_random_t, eval_freq, eval_eps,
#               save_freq, replay_buffer, sampler, datatype, device, save_dir=save_dir,
#               save_episode=False, plot_function=plot_episode)


def configure(
        env_name_, num_envs_, gmaze_frame_skip_, gmaze_walls_,
        episode_max_length_, buffer_name_, buffer_size_,
        sampler_name_, goalenv_sampler_name_, agent_name_,
        seed_=None
):
    if env_name_.startswith('brax-'):
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_true_name = re.sub('-v.$', '', env_name_).removeprefix('brax-')
        gym_name = env_name_
        if gym_name not in gym.envs.registry.env_specs:
            entry_point = functools.partial(envs.create_gym_env, env_name=env_true_name)
            gym.register(gym_name, entry_point=entry_point)
        env_ = gym.make(gym_name, batch_size=num_envs_,
                        episode_length=episode_max_length_)
        # automatically convert between jax ndarrays and torch tensors:
        env_ = to_torch.JaxToTorchWrapper(env_, device=device_)
        datatype_ = xpag.tl.DataType.TORCH
    elif env_name_.startswith('GMaze'):
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_ = gym.make("GMazeSimple-v0",
                        device=device_,
                        batch_size=num_envs_,
                        frame_skip=gmaze_frame_skip_,
                        walls=gmaze_walls_)
        datatype_ = xpag.tl.DataType.TORCH
    else:
        if num_envs > 1:
            print('ok')
            env_ = gym.vector.make(env_name_, num_envs=num_envs)
            env_.spec = gym.envs.registration.EnvSpec(env_name_)
        else:
            env_ = gym.make(env_name_)
        datatype_ = xpag.tl.DataType.NUMPY
        device_ = 'cpu'

    agent_params = {}
    # Set seeds
    if seed_ is not None:
        env_.seed(seed_)
        torch.manual_seed(seed_)
        np.random.seed(seed_)
        agent_params['seed'] = seed_

    is_goalenv = xpag.tl.check_goalenv(env_)
    dimensions = xpag.tl.get_dimensions(env_)

    if buffer_name_ == 'DefaultBuffer':
        replay_buffer_ = xpag.tl.default_replay_buffer(
            buffer_size_,
            episode_max_length_,
            env_,
            datatype_,
            device_
        )
    else:
        replay_buffer_ = None  # only one available buffer so far

    if is_goalenv:
        sampler_ = eval('xpag.sa.' + goalenv_sampler_name_)(
            env.compute_reward, datatype=datatype_)
    else:
        sampler_ = eval('xpag.sa.' + sampler_name_)(datatype=datatype_)

    if is_goalenv:
        agent_ = eval('xpag.ag.' + agent_name_)(
            dimensions['observation_dim'] + dimensions['action_dim'],
            dimensions['action_dim'],
            params=agent_params)
    else:
        agent_ = eval('xpag.ag.' + agent_name_)(dimensions['observation_dim'],
                                                dimensions['action_dim'],
                                                params=agent_params)

    return agent_, env_, replay_buffer_, sampler_, datatype_, device_


gmaze_frame_skip = 2  # only used by gym-gmazes environments
gmaze_walls = []  # only used by gym-gmazes environments
env_name = 'HalfCheetah-v3'
# env_name = 'brax-halfcheetah-v0'
num_envs = 1
episode_max_length = 1000
buffer_name = 'DefaultBuffer'
buffer_size = 1e6
sampler_name = 'DefaultSampler'
goalenv_sampler_name = 'HER'  # only for environments with goals
agent_name = 'SAC'
seed = 0

agent, env, replay_buffer, sampler, datatype, device = configure(
    env_name, num_envs, gmaze_frame_skip, gmaze_walls, episode_max_length, buffer_name,
    buffer_size, sampler_name, goalenv_sampler_name, agent_name, seed
)

save_dir = os.path.join(os.path.expanduser("~"),
                        "results",
                        "xpag",
                        datetime.now().strftime("%Y%m%d_%H%M%S"))

plot_episode = functools.partial(
    plot_episode_2d,
    plot_env_function=env.plot if hasattr(env, "plot") else None
)
plot_episode = None
max_t = int(1e6)
train_ratio = 1.
batch_size = 256
start_random_t = 0
eval_freq = 1000 * 5
eval_eps = 5
save_freq = 0

xpag.tl.learn(agent, env, num_envs, episode_max_length,
              max_t, train_ratio, batch_size, start_random_t, eval_freq, eval_eps,
              save_freq, replay_buffer, sampler, datatype, device, save_dir=save_dir,
              save_episode=False, plot_function=plot_episode)
