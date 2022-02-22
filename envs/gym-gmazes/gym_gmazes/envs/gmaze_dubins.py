# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

# from abc import ABC
from abc import abstractmethod
from typing import Optional
import gym
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc


class GoalEnv(gym.Env):
    """The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key.'.format(key))

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'],
                                                    ob['desired_goal'], info)
        """
        raise NotImplementedError


def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
    y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

    criterion1 = denom != 0
    t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
    criterion2 = torch.logical_and(t > 0, t < 1)
    t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
    criterion3 = torch.logical_and(t > 0, t < 1)

    return torch.logical_and(torch.logical_and(criterion1, criterion2), criterion3)


class GMazeCommon:
    def __init__(self, device: str = "cpu", batch_size: int = 1):
        self.batch_size = batch_size
        self.device = device
        utils.EzPickle.__init__(**locals())
        self.reward_function = None
        self.frame_skip = 2

        # initial position + orientation
        self.init_qpos = torch.tensor(
            np.tile(np.array([-1.0, 0.0, 0.0]), (self.batch_size, 1))
        ).to(self.device)
        self.init_qvel = None  # velocities are not used
        self.state = self.init_qpos
        self.walls = []
        self._obs_dim = 3
        self._action_dim = 1
        self.num_steps = 0
        high = np.tile(1.0 * np.ones(self._action_dim), (self.batch_size, 1))
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def set_reward_function(self, reward_function):
        self.reward_function = (
            reward_function  # the reward function is not defined by the environment
        )

    def set_frame_skip(self, frame_skip: int = 2):
        self.frame_skip = (
            frame_skip  # a call to step() repeats the action frame_skip times
        )

    def set_walls(self, walls=None):
        if walls is None:
            self.walls = [
                ([0.5, -0.5], [0.5, 1.01]),
                ([-0.5, -0.5], [-0.5, 1.01]),
                ([0.0, -1.01], [0.0, 0.5]),
            ]
        else:
            self.walls = walls

    def plot(self, ax):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])


def default_reward_fun(action, new_obs):
    # reward = 1. * (torch.logical_and(new_obs[:, 0] > 0.5, new_obs[:, 1] > 0.5))
    reward = 1.0 * (torch.logical_and(new_obs[:, 0] > -0.5, new_obs[:, 1] > 0.0))
    return torch.unsqueeze(reward, dim=-1)
