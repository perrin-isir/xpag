import os
import torch
import numpy as np
from xpag.agents.agent import Agent
from xpag.buffers.buffer import Buffer
from xpag.samplers.sampler import Sampler
from xpag.tools.utils import DataType, define_step_data, \
    step_data_select, reshape_func, hstack_func, max_func, datatype_convert, \
    register_step_in_episode
import gym
from xpag.buffers.buffer import DefaultBuffer

from IPython import embed


class SaveEpisode:
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.type = 'brax' if env.spec.id.startswith('brax') else 'mujoco'
        self.qpos = []
        self.qrot = []
        self.qvel = []
        self.qang = []

    def update(self):
        if self.type == 'brax':
            self.qpos.append(self.env.unwrapped._state.qp.pos.to_py())
            self.qrot.append(self.env.unwrapped._state.qp.rot.to_py())
            self.qvel.append(self.env.unwrapped._state.qp.vel.to_py())
            self.qang.append(self.env.unwrapped._state.qp.ang.to_py())
        else:
            posvel = np.split(reshape_func(self.env.state_vector(), (1, -1)),
                              [self.env.init_qpos.shape[-1]], axis=1)
            self.qpos.append(posvel[0])
            self.qvel.append(posvel[1])

    def save(self, i: int, save_dir: str):
        os.makedirs(os.path.join(save_dir, 'episode'), exist_ok=True)
        if self.type == 'brax':
            env_name = [self.env.spec.id.removeprefix('brax-').removesuffix('-v0')]
            np.savetxt(os.path.join(save_dir, 'episode', 'env_name.txt'),
                       env_name, delimiter='\n', fmt="%s")
            np.save(os.path.join(save_dir, 'episode', 'qp_pos'),
                    [pos[i] for pos in self.qpos])
            np.save(os.path.join(save_dir, 'episode', 'qp_rot'),
                    [rot[i] for rot in self.qrot])
            np.save(os.path.join(save_dir, 'episode', 'qp_vel'),
                    [vel[i] for vel in self.qvel])
            np.save(os.path.join(save_dir, 'episode', 'qp_ang'),
                    [ang[i] for ang in self.qang])
        else:
            env_name = [self.env.spec.id]
            np.savetxt(os.path.join(save_dir, 'episode', 'env_name.txt'),
                       env_name, delimiter='\n', fmt="%s")
            np.save(os.path.join(save_dir, 'episode', 'qpos'),
                    [pos[i] for pos in self.qpos])
            np.save(os.path.join(save_dir, 'episode', 'qvel'),
                    [vel[i] for vel in self.qvel])

        self.qpos = []
        self.qrot = []
        self.qvel = []
        self.qang = []


def check_goalenv(env) -> bool:
    if isinstance(env, gym.Wrapper):
        env_class = env.unwrapped.__class__
    else:
        env_class = env.__class__
    return issubclass(env_class, gym.core.GoalEnv)


def get_dimensions(env) -> dict:
    is_goalenv = check_goalenv(env)
    if hasattr(env, "is_vector_env"):
        gym_vec_env = env.is_vector_env
    else:
        gym_vec_env = False
    dims = {}
    if gym_vec_env:
        dims['action_dim'] = env.single_action_space.shape[-1]
        dims['observation_dim'] = env.single_observation_space['observation'].shape[
            -1] if is_goalenv else env.single_observation_space.shape[-1]
        dims['achieved_goal_dim'] = env.single_observation_space['achieved_goal'].shape[
            -1] if is_goalenv else None
        dims['desired_goal_dim'] = env.single_observation_space['desired_goal'].shape[
            -1] if is_goalenv else None
    else:
        dims['action_dim'] = env.action_space.shape[-1]
        dims['observation_dim'] = env.observation_space['observation'].shape[
            -1] if is_goalenv else env.observation_space.shape[-1]
        dims['achieved_goal_dim'] = env.observation_space['achieved_goal'].shape[
            -1] if is_goalenv else None
        dims['desired_goal_dim'] = env.observation_space['desired_goal'].shape[
            -1] if is_goalenv else None
    return dims


def default_replay_buffer(buffer_size: int, episode_max_length: int,
                          env, datatype: DataType, device: str = 'cpu'):
    is_goalenv = check_goalenv(env)
    dims = get_dimensions(env)
    action_dim = dims['action_dim']
    observation_dim = dims['observation_dim']
    achieved_goal_dim = dims['achieved_goal_dim']
    desired_goal_dim = dims['desired_goal_dim']
    if is_goalenv:
        replay_buffer = DefaultBuffer(
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
            buffer_size,
            datatype=datatype,
            device=device,
        )
    else:
        replay_buffer = DefaultBuffer(
            {
                'obs': observation_dim,
                'obs_next': observation_dim,
                'actions': action_dim,
                'r': 1,
                'terminals': 1,
            },
            episode_max_length,
            buffer_size,
            datatype=datatype,
            device=device,
        )
    return replay_buffer


def learn(
        agent: Agent,
        env,
        num_envs: int,
        episode_max_length: int,
        max_t: int,
        train_ratio: float,
        batch_size: int,
        start_random_t: int,
        eval_freq: int,
        eval_episodes: int,
        save_freq: int,
        replay_buffer: Buffer,
        sampler: Sampler,
        datatype: DataType,
        device: str,
        save_dir: str = None,
        save_episode: bool = False,
        plot_function=None,
):
    def init_done(value: float):
        if datatype == DataType.TORCH:
            return value * torch.ones(num_envs, device=device)
        elif datatype == DataType.NUMPY:
            return value * np.ones(num_envs)

    if save_episode:
        save_ep = SaveEpisode(env, num_envs)
    else:
        save_ep = None

    is_goalenv = check_goalenv(env)
    dimensions = get_dimensions(env)

    StepDataUnique, StepDataMultiple = define_step_data(
        is_goalenv, num_envs, dimensions['observation_dim'],
        dimensions['achieved_goal_dim'], dimensions['desired_goal_dim'],
        dimensions['action_dim'], episode_max_length,
        datatype, device)

    total_t = 0
    t_since_eval = 0
    t_since_save = 0
    episode_num = 0
    episode_mean_reward = 0
    episode_rewards = datatype_convert(np.zeros((num_envs, 1)), datatype, device)
    episode_success = datatype_convert(np.zeros((num_envs, 1)), datatype, device)
    episode_t = 0
    episode = None
    eval_ep = None
    avg_reward = None
    episode_argmax = None
    o = None
    done = init_done(1)

    mode = 'training'
    while total_t < max_t:

        # as soon as one episode is done we terminate all the episodes
        if done.max() or episode_t >= episode_max_length:

            if total_t != 0:

                # save episode in the buffer and train the agent
                if mode == 'training':
                    index = episode_rewards.argmax()
                    # index = episode_success.argmax()
                    step_data_select(
                        episode_argmax, episode, index
                    )
                    print(f'[{episode_num}] best episode out of {num_envs}: ' +
                          f'reward = {episode_rewards.max()}')

                    if save_dir is not None and save_episode:
                        save_ep.save(index, save_dir)

                    if save_dir is not None and plot_function is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        plot_function(
                            os.path.join(save_dir,
                                         'ts{:010d}.png'.format(total_t)),
                            episode_argmax,
                            # episode,
                            episode_t)
                    replay_buffer.store_episode(1, episode_argmax, episode_t)
                    # replay_buffer.store_episode(num_envs, episode, episode_t)

                    for _ in range(int(train_ratio * episode_t)):
                        pre_sample = replay_buffer.pre_sample()
                        # embed()
                        # quit()
                        agent.train(pre_sample, sampler, batch_size)

                if mode == 'eval':
                    eval_ep -= 1
                    avg_reward += episode_mean_reward
                    if eval_ep <= 0:
                        eval_ep = eval_episodes
                        avg_reward /= eval_episodes
                        print(f'[EVAL] average reward over {eval_episodes} ' +
                              f'x {num_envs} episode(s): {avg_reward}')
                        if is_goalenv:
                            print(f'[EVAL] average success rate over {eval_episodes} ' +
                                  f'x {num_envs} episode(s): {episode_success.mean()}')
                        mode = 'training'
                else:
                    episode_num += 1

                # switch to 'eval' mode
                if t_since_eval >= eval_freq:
                    t_since_eval %= eval_freq
                    mode = 'eval'
                    eval_ep = eval_episodes
                    avg_reward = 0.

            # env reset
            o = env.reset()

            if save_episode:
                save_ep.update()

            init_done(0)
            episode_mean_reward = 0
            episode_rewards *= 0
            episode_success *= 0
            episode_t = 0
            episode = StepDataMultiple()
            episode_argmax = StepDataUnique()

        # select action randomly or according to policy
        if total_t < start_random_t and mode == 'training':
            action = datatype_convert(
                env.action_space.sample(),
                datatype,
                device)
        else:
            deter = True if mode == 'eval' else False
            if is_goalenv:
                action = datatype_convert(
                    agent.select_action(
                        hstack_func(o["observation"], o["desired_goal"]),
                        deterministic=False,
                    ),
                    datatype,
                    device)
            else:
                action = datatype_convert(
                    agent.select_action(o, deterministic=deter),
                    datatype,
                    device)

        new_o, reward, done, info = env.step(action)
        if save_episode:
            save_ep.update()

        reward = reshape_func(reward, (num_envs, 1))
        done = reshape_func(done, (num_envs, 1))

        register_step_in_episode(
            episode,
            episode_t,
            is_goalenv,
            num_envs,
            o,
            action,
            new_o,
            reward,
            done
        )

        if is_goalenv:
            episode_success = max_func(episode_success, reshape_func(
                info['is_success'], (num_envs, 1)))

        episode_mean_reward += reward.mean()
        episode_rewards += reward
        episode_t += 1
        o = new_o

        if mode == 'training':
            t_since_eval += 1
            t_since_save += 1
            total_t += 1
