

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import SAC
from xpag.tools import learn
from xpag.tools import mujoco_notebook_replay



def main():
    num_envs = 10  # the number of rollouts in parallel during training
    env, eval_env, env_info = gym_vec_env('HalfCheetah-v4', num_envs)

    batch_size = 256
    gd_steps_per_step = 1
    start_training_after_x_steps = env_info['max_episode_steps'] * 10
    max_steps = 10_000_000
    evaluate_every_x_steps = 5_000
    save_agent_every_x_steps = 100_000
    save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'train_mujoco')
    save_episode = True
    plot_projection = None

    agent = SAC(
    env_info['observation_dim'] if not env_info['is_goalenv']
    else env_info['observation_dim'] + env_info['desired_goal_dim'],
    env_info['action_dim'],
    {}
    )   
    sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER(env.compute_reward)
    buffer = DefaultEpisodicBuffer(
        max_episode_steps=env_info['max_episode_steps'],
        buffer_size=1_000_000,
        sampler=sampler
    )
    goalsetter = DefaultGoalSetter()

    learn(
    env,
    eval_env,
    env_info,
    agent,
    buffer,
    goalsetter,
    batch_size,
    gd_steps_per_step,
    start_training_after_x_steps,
    max_steps,
    evaluate_every_x_steps,
    save_agent_every_x_steps,
    save_dir,
    save_episode,
    plot_projection,
)

main()