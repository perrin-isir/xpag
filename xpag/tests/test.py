import xpag
from IPython import embed
import numpy as np
import torch
import gym
import string
import collections
import functools
import random
import time
import datetime
import brax
from brax import envs
from brax.envs import to_torch
from brax.io import metrics
from IPython import embed
import argparse
import os
import numpy as np
import logging

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device='cuda')
print(torch.cuda.memory_allocated(device='cuda'))


if False:
    import sys
    import subprocess

    preload = os.environ.get("LD_PRELOAD", "")
    if os.environ["HOME"] == "/home/perrin":
        if not preload:
            to_preload = "/usr/lib/x86_64-linux-gnu/libGLEW.so"
            os.environ["LD_PRELOAD"] = to_preload
            print("Restarting with LD_PRELOAD={0}".format(to_preload))
            os.execv(sys.executable, [sys.executable] + sys.argv)
    glxinfo = subprocess.Popen("glxinfo", stdout=subprocess.PIPE)
    output_glxinfo = glxinfo.communicate()[0]
    for line in output_glxinfo.decode("utf-8").split("\n"):
        if (
            "GLX version" in line
            or "OpenGL vendor string" in line
            or "OpenGL renderer string" in line
            or "OpenGL core profile version" in line
        ):
            print(line)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
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


def evaluate_agent(envir, agent_, eval_episodes=10):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        o_ = envir.reset()
        done_ = False
        while not done_:
            o_, r_, done_, _ = envir.step(
                agent_.select_action(o_, deterministic=True)
            )
            avg_reward += r_

    avg_reward /= eval_episodes
    print(
        '-----------------------------------------------------------------------'
    )
    print('Evaluation over %d episodes: %f' % (eval_episodes, avg_reward))
    print(
        '-----------------------------------------------------------------------'
    )
    return [eval_episodes, avg_reward, None]


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_max_length = 1000
num_envs = 1
env = gym.make('HalfCheetah-v3')
version = 'numpy'


if '_max_episode_steps' in dir(env):
    max_episode_steps = env._max_episode_steps
else:
    max_episode_steps = 1000

# Set seeds
if args.seed is not None:
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

action_dim = env.action_space.shape[-1]
max_action = env.action_space.high

observation_dim = env.observation_space.shape[-1]
o_dim = observation_dim
params = {'max_action': max_action, 'device': device}

datatype = xpag.tl.DataType.NUMPY

replay_buffer = xpag.bf.DefaultBuffer(
    {
        'obs': observation_dim,
        'obs_next': observation_dim,
        'actions': action_dim,
        'r': 1,
        'terminals': 1,
    },
    max_episode_steps,
    args.buffer_size,
    datatype=datatype,
    device=device,
)
sampler = xpag.sa.DefaultSampler(datatype)

agent = xpag.ag.SAC(o_dim, action_dim, device, params)

StepData = collections.namedtuple(
    'StepData',
    ('obs', 'obs_next', 'actions', 'r', 'terminals'))

episode_max_length = max_episode_steps


def define_step_data(num_envs: int, version: str = 'torch', device: str = 'cpu'):
    fields = ('obs', 'obs_next', 'actions', 'r', 'terminals')
    sizes = [o_dim, o_dim, action_dim, 1, 1]
    empty_func = functools.partial(torch.empty, device=device) if version == 'torch' \
        else np.empty

    def defaults(k):
        return [empty_func((k, episode_max_length, siz)) for siz in sizes]

    return collections.namedtuple('StepDataUnique', fields, defaults=defaults(1)), \
           collections.namedtuple('StepDataMultiple', fields,
                                  defaults=defaults(num_envs))


def step_data_select(sd_one, sd_m, i: int):
    for f in sd_one._fields:
        sd_one._asdict()[f][0] = sd_m._asdict()[f][i]
    # ep_argmax.obs[0] = ep.obs[i]
    # ep_argmax.obs_next[0] = ep.obs_next[i]
    # ep_argmax.actions[0] = ep.actions[i]
    # ep_argmax.r[0] = ep.r[i]
    # ep_argmax.terminals[0] = ep.terminals[i]


StepDataUnique, StepDataMultiple = define_step_data(num_envs, version, device)


def reshape_func(x, shape):
    if type(x) == torch.Tensor:
        return x.reshape(shape)
    elif type(x) == np.ndarray:
        return x.reshape(shape)
    else:
        return np.array(x).reshape(shape)


def version_convert(x, version='torch', device='cpu'):
    if version == 'torch':
        if type(x) == torch.Tensor:
            return x.to(device=device)
        else:
            return torch.tensor(x, device=device)
    else:
        if type(x) == torch.Tensor:
            return x.detach().cpu().numpy()
        else:
            return np.array(x)


# def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
#     """Map a function over each field in StepData."""
#     items = {}
#     keys = sds[0]._asdict().keys()
#     for k in keys:
#         items[k] = f(*[sd._asdict()[k] for sd in sds])
#     return StepData(**items)


# def empty_episode(nenv, version='torch', device=None):
#     if version == 'torch':
#         return StepData(
#             torch.empty((nenv, episode_max_length, o_dim), device=device),
#             torch.empty((nenv, episode_max_length, o_dim), device=device),
#             torch.empty((nenv, episode_max_length, action_dim), device=device),
#             torch.empty((nenv, episode_max_length, 1), device=device),
#             torch.empty((nenv, episode_max_length, 1), device=device),
#         )
#     else:
#         return StepData(
#             np.empty((nenv, episode_max_length, o_dim)),
#             np.empty((nenv, episode_max_length, o_dim)),
#             np.empty((nenv, episode_max_length, action_dim)),
#             np.empty((nenv, episode_max_length, 1)),
#             np.empty((nenv, episode_max_length, 1)),
#         )

def init_done(value: float, nenv: int, version : str='torch', device: str='cpu'):
    if version == 'torch':
        return value * torch.ones(nenv, device=device)
    else:
        return value * np.ones(nenv)


evaluations = []
total_timesteps = 0
timesteps_since_eval = 5000
timesteps_since_save = 0
episode_num = 0
episode_mean_reward = 0
episode_rewards = version_convert(np.zeros((num_envs, 1)), version, device)
episode_t = 0
trajectory = []
episode = None
episode_argmax = None
o = None
tic = None
done = init_done(1, num_envs, version, device)

while total_timesteps < args.max_timesteps:

    # As soon as one episode is done we terminate all the episodes
    if done.max() or episode_t >= max_episode_steps:

        if total_timesteps != 0:
            step_data_select(episode_argmax, episode, episode_rewards.argmax())
            # embed()
            print(episode_num, episode_rewards.max())
            # replay_buffer.store_episode(trajectory)
            replay_buffer.store_episode(1, episode_argmax, episode_t)
            # replay_buffer.store_episode(num_envs, episode, episode_t)

            for _ in range(int(args.train_ratio * episode_t)):
                pre_sample = replay_buffer.pre_sample()
                agent.train(pre_sample, sampler, args.batch_size)

        # Evaluate episode and save
        # if timesteps_since_eval >= args.eval_freq:
        #     timesteps_since_eval %= args.eval_freq
        #     eval_episodes = 10
        #     eval_result = evaluate_agent(env, agent, eval_episodes)
        #     evaluations.append(eval_result)

        # Env reset
        o = env.reset()
        # Active exploration with the goal setter:

        init_done(0, num_envs, version, device)
        episode_mean_reward = 0
        episode_rewards *= 0
        episode_t = 0
        episode_num += 1
        trajectory = []
        # episode = empty_episode(num_envs, version, device)
        episode = StepDataMultiple()
        # episode_argmax = empty_episode(1, version, device)
        episode_argmax = StepDataUnique()

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = version_convert(env.action_space.sample(),
                                 version, device)
        # action = env.action_space.sample()
    else:
        action = version_convert(agent.select_action(o, deterministic=False),
                                 version, device)

    # from IPython import embed as e; e(); quit()
    new_o, reward, done, info = env.step(action)

    # Append data to trajectory
    # trajectory.append(
    #     {
    #         'obs': o,
    #         'obs_next': new_o,
    #         'actions': action,
    #         'r': reward,
    #         'terminals': float(done),
    #     }
    # )
    # embed()

    reward = reshape_func(reward, (num_envs, 1))
    done = reshape_func(done, (num_envs, 1))
    episode.r[:, episode_t, :] = reward
    episode.terminals[:, episode_t, :] = done
    episode.obs[:, episode_t, :] = reshape_func(o, (num_envs, observation_dim))
    episode.obs_next[:, episode_t, :] = reshape_func(new_o, (num_envs, observation_dim))
    episode.actions[:, episode_t, :] = reshape_func(action, (num_envs, action_dim))

    # episode.obs[:, episode_t, :] = o.reshape((num_envs, observation_dim))
    # episode.obs_next[:, episode_t, :] = new_o.reshape((num_envs, observation_dim))
    # episode.actions[:, episode_t, :] = action.reshape((num_envs, action_dim))
    # episode.r[:, episode_t, :] = reward.reshape((num_envs, 1))
    # episode.terminals[:, episode_t, :] = done.reshape((num_envs, 1))

    episode_mean_reward += reward.mean()
    episode_rewards += reward
    episode_t += 1
    total_timesteps += 1
    timesteps_since_eval += 1
    timesteps_since_save += 1

    # if episode_t >= max_episode_steps:
    #     done = True

    o = new_o
