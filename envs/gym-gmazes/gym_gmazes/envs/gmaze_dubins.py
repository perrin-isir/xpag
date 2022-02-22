# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
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


class GMazeDubins(GMazeCommon, gym.Env, utils.EzPickle, ABC):
    def __init__(self, device: str = "cpu", batch_size: int = 1):
        super().__init__(device, batch_size)

        self.set_reward_function(default_reward_fun)

        high = np.tile(1.0 * np.ones(self._obs_dim), (self.batch_size, 1))
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

    def step(self, action: torch.Tensor):
        # add action to the state frame_skip times,
        # checking -1 & +1 boundaries and intersections with walls
        for k in range(self.frame_skip):
            cosval = torch.cos(torch.pi * self.state[:, 2])
            sinval = torch.sin(torch.pi * self.state[:, 2])
            ns_01 = self.state[:, :2] + 1.0 / 20.0 * torch.stack(
                (cosval, sinval), dim=1
            ).to(self.device)
            ns_01 = ns_01.clip(-1.0, 1.0)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.0
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = torch.hstack((ns_01, ns_2.unsqueeze(dim=1)))

            intersection = torch.full((self.batch_size,), False).to(self.device)
            for (w1, w2) in self.walls:
                intersection = torch.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2)
                )
            intersection = torch.unsqueeze(intersection, dim=-1)
            self.state = self.state * intersection + new_state * torch.logical_not(
                intersection
            )

        observation = self.state
        reward = self.reward_function(action, observation)
        self.num_steps += 1
        done = torch.full((self.batch_size, 1), False).to(self.device)
        info = None  # no info
        return observation, reward, done, info

    def reset_model(self):
        # reset state to initial value
        self.state = self.init_qpos

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()
        self.num_steps = 0
        return self.state

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor = None):
        self.state = qpos


def achieved_g(state):
    s1 = state[:, :2]
    # s2 = (s1 / (1 / 3.)).int() / 3.
    # s3 = (s1 / (1 / 2.)).int() / 2.
    # return torch.hstack((s1, s2, s3))
    return s1
    # return (s1 / (1 / 3.)).int() / 3.


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return torch.linalg.norm(goal_a[:, :2] - goal_b[:, :2], axis=-1)


def default_compute_reward(
    achieved_goal: torch.Tensor, desired_goal: torch.Tensor, info: dict
):
    distance_threshold = 0.1
    reward_type = "sparse"
    d = goal_distance(achieved_goal, desired_goal)
    if reward_type == "sparse":
        return -1.0 * (d > distance_threshold)
    else:
        return -d


def default_success_function(achieved_goal: torch.Tensor, desired_goal: torch.Tensor):
    distance_threshold = 0.1
    d = goal_distance(achieved_goal, desired_goal)
    return 1.0 * (d < distance_threshold)


class GMazeGoalDubins(GMazeCommon, GoalEnv, utils.EzPickle, ABC):
    def __init__(self, device: str = "cpu", batch_size: int = 1):
        super().__init__(device, batch_size)

        high = np.tile(1.0 * np.ones(self._obs_dim), (self.batch_size, 1))
        low = -high
        self._achieved_goal_dim = 2
        self._desired_goal_dim = 2
        high_achieved_goal = np.tile(
            1.0 * np.ones(self._achieved_goal_dim), (self.batch_size, 1)
        )
        low_achieved_goal = -high_achieved_goal
        high_desired_goal = np.tile(
            1.0 * np.ones(self._desired_goal_dim), (self.batch_size, 1)
        )
        low_desired_goal = -high_desired_goal
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low, high, dtype=np.float64),
                achieved_goal=spaces.Box(
                    low_achieved_goal, high_achieved_goal, dtype=np.float64
                ),
                desired_goal=spaces.Box(
                    low_desired_goal, high_desired_goal, dtype=np.float64
                ),
            )
        )
        self.goal = None

        self.compute_reward = None
        self.set_reward_function(default_compute_reward)

        self._is_success = None
        self.set_success_function(default_success_function)

    def set_reward_function(self, reward_function):
        self.compute_reward = (  # the name is compute_reward in GoalEnv environments
            reward_function
        )

    def set_success_function(self, success_function):
        self._is_success = success_function

    def _sample_goal(self):
        # return (torch.rand(self.batch_size, 2) * 2. - 1).to(self.device)
        return achieved_g(torch.rand(self.batch_size, 2) * 2.0 - 1).to(self.device)

    def reset_model(self):
        # reset state to initial value
        self.state = self.init_qpos

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()  # reset state to initial value
        self.goal = self._sample_goal()  # sample goal
        self.num_steps = 0
        return {
            "observation": self.state,
            "achieved_goal": achieved_g(self.state),
            "desired_goal": self.goal,
        }

    def step(self, action: torch.Tensor):
        # add action to the state frame_skip times,
        # checking -1 and +1 boundaries and intersections with walls
        for k in range(self.frame_skip):
            cosval = torch.cos(torch.pi * self.state[:, 2])
            sinval = torch.sin(torch.pi * self.state[:, 2])
            ns_01 = self.state[:, :2] + 1.0 / 20.0 * torch.stack(
                (cosval, sinval), dim=1
            ).to(self.device)
            ns_01 = ns_01.clip(-1.0, 1.0)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.0
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = torch.hstack((ns_01, ns_2.unsqueeze(dim=1)))

            intersection = torch.full((self.batch_size,), False).to(self.device)
            for (w1, w2) in self.walls:
                intersection = torch.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2)
                )
            intersection = torch.unsqueeze(intersection, dim=-1)
            self.state = self.state * intersection + new_state * torch.logical_not(
                intersection
            )

        reward = self.compute_reward(achieved_g(self.state), self.goal, {})
        self.num_steps += 1

        # done = torch.full((self.batch_size, 1), False).to(self.device)
        info = {"is_success": self._is_success(achieved_g(self.state), self.goal)}
        done = info["is_success"].reshape((self.batch_size, 1))

        return (
            {
                "observation": self.state,
                "achieved_goal": achieved_g(self.state),
                "desired_goal": self.goal,
            },
            reward,
            done,
            info,
        )
