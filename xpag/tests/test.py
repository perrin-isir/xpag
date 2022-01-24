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
from brax import envs
from brax.envs import to_torch
from brax.io import metrics
from IPython import embed
import argparse
import os
import logging
import xpag
from xpag.plotting.basics import plot_episode_2d

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
print(torch.cuda.memory_allocated(device='cuda'))


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(rnddir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="t01")
    parser.add_argument(
        "--env_name",
        default="InvertedPendulum-v2"
        # default="SimpleMazeGoalEnv-v0"
        # default="HalfCheetahBulletEnv-v2",
    )  # OpenAI gym environment name
    parser.add_argument("--agent_name", default="TD3")  # Policy name
    parser.add_argument("--device", default=None)  # "cuda" or "cpu"
    parser.add_argument("--sampler_name", default="DefaultSampler")
    parser.add_argument("--buffer_name", default="DefaultBuffer")
    parser.add_argument("--goalsetter_name", default="DefaultGoalSetter")
    parser.add_argument("--seed", type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--load", type=str
    )  # Load agent, goal setter and buffer with a given ID

    parser.add_argument(
        "--render", type=str2bool, nargs="?", const=True, default=False
    )  # Activate rendering
    parser.add_argument(
        "--realtime", type=str2bool, nargs="?", const=True, default=False
    )  # If True, episodes are rendered at 60 hz
    parser.add_argument(
        "--plotpaths", type=str2bool, nargs="?", const=True, default=False
    )  # If True, during training every episode is plotted and saved in a .png file
    parser.add_argument(
        "--start_timesteps", default=10_000, type=int
    )  # How many time steps purely random policy is run for
    parser.add_argument(
        "--eval_freq", default=5000, type=float
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--save_freq", default=np.inf, type=float
    )  # How often (time steps) models and data (agent + goal setter + buffer) are saved
    parser.add_argument(
        "--max_timesteps", default=1e6, type=float
    )  # Max time steps to run environment for
    parser.add_argument(
        "--buffer_size", default=1e6, type=float
    )  # Replay buffer size
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument(
        "--train_ratio", default=1.0, type=float
    )  # Number of training batches per step
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.expanduser("~"), "results", "SGE_experiments"),
    )
    parser.add_argument(
        "--save_dir",
        default="save_dir",
    )
    args = parser.parse_args()
    args.save_dir = os.path.expanduser(
        os.path.join(args.output_dir, args.save_dir, args.env_name, args.tag, rnddir)
    )
    return args


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.__level = level

    def filter(self, logrecord):
        return logrecord.levelno <= self.__level


def log_init(args, agent, gsetter, init_list, init_list_test):
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.txt"), "w") as f:
        print("last commit:", file=f)
        print(os.popen("git rev-parse --short HEAD").read()[:-1], file=f)
        print("\nargs: ", file=f)
        print(args, file=f)
        print("\nAgent config:", file=f)
        agent.write_config(f)
        print("\nGoalSetter parameters:", file=f)
        gsetter.write_params(f)
        f.close()
    logger = logging.getLogger("SGE-logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fhfilter = LevelFilter(logging.INFO)
    fh.addFilter(fhfilter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fhformatter = logging.Formatter("%(message)s")
    chformatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(fhformatter)
    ch.setFormatter(chformatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(",".join(map(str, init_list)))

    logger_test = logging.getLogger("SGE-logger-test")
    logger_test.setLevel(logging.INFO)
    fh_test = logging.FileHandler(os.path.join(args.save_dir, "log_test.txt"))
    fh_test.setLevel(logging.INFO)
    fhfilter_test = LevelFilter(logging.INFO)
    fh_test.addFilter(fhfilter_test)
    ch_test = logging.StreamHandler()
    ch_test.setLevel(logging.WARNING)
    fhformatter_test = logging.Formatter("%(message)s")
    chformatter_test = logging.Formatter("%(asctime)s ------------ TEST: %(message)s")
    fh_test.setFormatter(fhformatter_test)
    ch_test.setFormatter(chformatter_test)
    logger_test.addHandler(fh_test)
    logger_test.addHandler(ch_test)
    logger_test.info(",".join(map(str, init_list_test)))

    return logger, logger_test


args = get_args('')

# env_name = 'halfcheetah'
# device = 'cuda'
# episode_max_length = 1000
# num_envs = 128
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

device = 'cuda'
num_envs = 1024
episode_max_length = 50
env = gym.make("GMazeGoalSimple-v0",
               device=device,
               batch_size=num_envs,
               frame_skip=2,
               walls=None)
datatype = xpag.tl.DataType.TORCH

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# episode_max_length = 1000
# num_envs = 1
# env = gym.make('HalfCheetah-v3')
# version = 'numpy'
# datatype = xpag.tl.DataType.NUMPY


# Set seeds
if args.seed is not None:
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

if isinstance(env, gym.Wrapper):
    env_class = env.unwrapped.__class__
else:
    env_class = env.__class__

is_goalenv = issubclass(env_class, gym.core.GoalEnv)

action_dim = env.action_space.shape[-1]
max_action = env.action_space.high

observation_dim = env.observation_space['observation'].shape[-1] if is_goalenv \
    else env.observation_space.shape[-1]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[-1] if is_goalenv \
    else None
desired_goal_dim = env.observation_space['desired_goal'].shape[-1] if is_goalenv \
    else None

if is_goalenv:
    replay_buffer = xpag.bf.DefaultBuffer(
        {
            "obs": observation_dim,
            "obs_next": observation_dim,
            "ag": achieved_goal_dim,
            "ag_next": achieved_goal_dim,
            "g": desired_goal_dim,
            "g_next": desired_goal_dim,
            "actions": action_dim,
            "terminals": 1,
        },
        episode_max_length,
        args.buffer_size,
        datatype=datatype,
        device=device,
    )
else:
    replay_buffer = xpag.bf.DefaultBuffer(
        {
            'obs': observation_dim,
            'obs_next': observation_dim,
            'actions': action_dim,
            'r': 1,
            'terminals': 1,
        },
        episode_max_length,
        args.buffer_size,
        datatype=datatype,
        device=device,
    )

if is_goalenv:
    sampler = xpag.sa.HER(env.compute_reward, datatype=datatype)
else:
    sampler = xpag.sa.DefaultSampler(datatype=datatype)

if is_goalenv:
    agent = xpag.ag.SAC(observation_dim + action_dim, action_dim, device, params=None)
else:
    agent = xpag.ag.SAC(observation_dim, action_dim, device, params=None)

save_dir = os.path.join(os.path.expanduser("~"),
                        "results",
                        "xpag",
                        datetime.now().strftime("%Y%m%d_%H%M%S"))

plot_episode = functools.partial(
    plot_episode_2d,
    plot_env_function=env.plot if hasattr(env, "plot") else None
)
# plot_episode = None
max_t = int(1e6)
train_ratio = 1.
batch_size = 256
start_random_t = 0
# eval_freq = 50 * 7
eval_freq = 1000 * 5
eval_episodes = 5
save_freq = 0

xpag.tl.learn(agent, env, num_envs, episode_max_length,
              max_t, train_ratio, batch_size, start_random_t, eval_freq, eval_episodes,
              save_freq, replay_buffer, sampler, datatype, device, save_dir=save_dir,
              plot_function=plot_episode)
