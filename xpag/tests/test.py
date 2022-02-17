import functools
from datetime import datetime
import os
import numpy as np
import torch
import xpag
from xpag.plotting.basics import plot_episode_2d

gmaze_frame_skip = 1  # only used by gym-gmazes environments
gmaze_walls = []  # only used by gym-gmazes environments
# env_name = 'HalfCheetah-v3'
# env_name = 'brax-halfcheetah-v0'
env_name = 'GMazeGoalDubins-v0'
num_envs = 2
# episode_max_length = 1000
episode_max_length = 50
buffer_name = 'DefaultBuffer'
buffer_size = 1e6
sampler_name = 'DefaultSampler'
goalenv_sampler_name = 'HER'  # only for environments with goals
agent_name = 'SAC'
# goalsetter_name = 'DefaultGoalSetter'
goalsetter_name = 'SGS'
seed = 0

agent, goalsetter, env, replay_buffer, sampler, datatype, device = xpag.tl.configure(
    env_name, num_envs, gmaze_frame_skip, gmaze_walls, episode_max_length, buffer_name,
    buffer_size, sampler_name, goalenv_sampler_name, agent_name, goalsetter_name, seed
)

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
train_ratio = num_envs * 1
batch_size = 256
start_random_t = int(np.ceil(episode_max_length * 5 / num_envs))
eval_freq = episode_max_length * 5
eval_eps = int(np.ceil(5 / num_envs))
save_freq = 0

goalsetter.set_sequence(
    [
        np.array([-0.5, -0.5]),
        np.array([0., 0.5]),
        np.array([0.5, -0.5])
    ],
    [20, 20, 20]
)

xpag.tl.learn(agent, goalsetter, env, num_envs, episode_max_length,
              max_t, train_ratio, batch_size, start_random_t, eval_freq, eval_eps,
              save_freq, replay_buffer, sampler, datatype, device, save_dir=save_dir,
              save_episode=False, plot_function=plot_episode)
