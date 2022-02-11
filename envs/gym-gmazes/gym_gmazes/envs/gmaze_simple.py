from abc import ABC
import gym
from gym import utils, spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
    y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

    criterion1 = (denom != 0)
    t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
    criterion2 = torch.logical_and(t > 0, t < 1)
    t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
    criterion3 = torch.logical_and(t > 0, t < 1)

    return torch.logical_and(torch.logical_and(criterion1, criterion2), criterion3)


class GMazeCommon:
    def __init__(
            self,
            device: str = 'cpu',
            batch_size: int = 1,
            frame_skip: int = 2,
            walls=None,
            reward_function=None,
            success_function=None
    ):
        self.batch_size = batch_size
        self.device = device
        utils.EzPickle.__init__(**locals())
        self.reward_function = (
            reward_function  # the reward function is not defined by the environment
        )
        self.success_function = success_function
        self.frame_skip = (
            frame_skip  # a call to step() repeats the action frame_skip times
        )

        # initial position + orientation
        # self.init_qpos = np.tile(np.array([-1., 0.]), (self.batch_size, 1))
        self.init_qpos = np.tile(np.array([-1., 0., 0.]), (self.batch_size, 1))
        self.init_qvel = []  # velocities are not used
        self.state = torch.tensor(self.init_qpos).to(self.device)
        if walls is None:
            self.walls = [
                ([0.5, -0.5], [0.5, 1.01]),
                ([-0.5, -0.5], [-0.5, 1.01]),
                ([0.0, -1.01], [0.0, 0.5]),
            ]
        else:
            self.walls = walls

        # self._obs_dim = 2
        self._obs_dim = 3
        # self._action_dim = 2
        self._action_dim = 1
        self.num_steps = 0
        high = np.tile(1.0 * np.ones(self._action_dim), (self.batch_size, 1))
        low = -high
        self.action_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def plot(self, ax):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)


def default_reward_fun(action, new_obs):
    reward = 1. * (torch.logical_and(new_obs[:, 0] > 0.5, new_obs[:, 1] > 0.5))
    return torch.unsqueeze(reward, dim=-1)


class GMazeSimple(GMazeCommon, gym.Env, utils.EzPickle, ABC):
    def __init__(
            self,
            device: str = 'cpu',
            batch_size: int = 1,
            frame_skip: int = 2,
            walls=None,
            reward_function=default_reward_fun,
    ):
        super().__init__(
            device, batch_size, frame_skip, walls, reward_function)

        high = np.tile(1.0 * np.ones(self._obs_dim), (self.batch_size, 1))
        low = -high
        self.observation_space = spaces.Box(
            low, high, dtype=np.float32
        )

    def step(self, action: torch.Tensor):
        # add action to the state frame_skip times,
        # checking -1 & +1 boundaries and intersections with walls
        for k in range(self.frame_skip):
            ns_01 = self.state[:, :2] + 1. / 20. * torch.stack(
                (
                    torch.cos(torch.pi * self.state[:, 2]),
                    torch.sin(torch.pi * self.state[:, 2])
                ), dim=1).to(self.device)
            ns_01 = ns_01.clip(-1., 1.)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = torch.hstack((ns_01, ns_2.unsqueeze(dim=1)))

            intersection = torch.full((self.batch_size,), False).to(self.device)
            for (w1, w2) in self.walls:
                intersection = torch.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2))
            intersection = torch.unsqueeze(intersection, dim=-1)
            self.state = self.state * intersection + new_state * torch.logical_not(
                intersection)

        observation = self.state
        reward = self.reward_function(
            action, observation
        )
        self.num_steps += 1
        done = torch.full((self.batch_size, 1), False).to(self.device)
        info = None  # no info
        return observation, reward, done, info

    def reset_model(self):
        # reset state to initial value
        self.state = torch.tensor(self.init_qpos).to(self.device)

    def reset(self):
        self.reset_model()
        self.num_steps = 0
        return self.state

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor = None):
        self.state = qpos


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return torch.linalg.norm(goal_a - goal_b, axis=-1)


def default_compute_reward(achieved_goal: torch.Tensor,
                           desired_goal: torch.Tensor,
                           info: dict):
    distance_threshold = 0.1
    reward_type = "sparse"
    d = goal_distance(achieved_goal, desired_goal)
    if reward_type == "sparse":
        return -1. * (d > distance_threshold)
    else:
        return -d


def default_success_function(achieved_goal: torch.Tensor,
                             desired_goal: torch.Tensor):
    distance_threshold = 0.1
    d = goal_distance(achieved_goal, desired_goal)
    return 1. * (d < distance_threshold)


class GMazeGoalSimple(GMazeCommon, gym.GoalEnv, utils.EzPickle, ABC):
    def __init__(
            self,
            device: str = 'cpu',
            batch_size: int = 1,
            frame_skip: int = 2,
            walls=None,
            reward_function=default_compute_reward,
            success_function=default_success_function
    ):
        super().__init__(
            device, batch_size, frame_skip, walls, reward_function)

        high = np.tile(1.0 * np.ones(self._obs_dim), (self.batch_size, 1))
        low = -high
        self._achieved_goal_dim = 2
        self._desired_goal_dim = 2
        high_achieved_goal = np.tile(1.0 * np.ones(self._achieved_goal_dim),
                                     (self.batch_size, 1))
        low_achieved_goal = -high_achieved_goal
        high_desired_goal = np.tile(1.0 * np.ones(self._desired_goal_dim),
                                    (self.batch_size, 1))
        low_desired_goal = -high_desired_goal
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    low, high, dtype=np.float32
                ),
                achieved_goal=spaces.Box(
                    low_achieved_goal, high_achieved_goal, dtype=np.float32
                ),
                desired_goal=spaces.Box(
                    low_desired_goal, high_desired_goal, dtype=np.float32
                ),
            )
        )
        self.goal = None
        self.compute_reward = reward_function
        self._is_success = success_function

    def _sample_goal(self):
        return (torch.rand(self.batch_size, 2) * 2. - 1).to(self.device)

    def reset_model(self):
        # reset state to initial value
        self.state = torch.tensor(self.init_qpos).to(self.device)

    def reset(self):
        self.reset_model()  # reset state to initial value
        self.goal = self._sample_goal()  # sample goal
        self.num_steps = 0
        return {
            "observation": self.state,
            "achieved_goal": self.state[:, :2],
            "desired_goal": self.goal,
        }

    def step(self, action: torch.Tensor):
        # add action to the state frame_skip times,
        # checking -1 & +1 boundaries and intersections with walls
        for k in range(self.frame_skip):
            ns_01 = self.state[:, :2] + 1. / 20. * torch.stack(
                (
                    torch.cos(torch.pi * self.state[:, 2]),
                    torch.sin(torch.pi * self.state[:, 2])
                ), dim=1).to(self.device)
            ns_01 = ns_01.clip(-1., 1.)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = torch.hstack((ns_01, ns_2.unsqueeze(dim=1)))

            intersection = torch.full((self.batch_size,), False).to(self.device)
            for (w1, w2) in self.walls:
                intersection = torch.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2))
            intersection = torch.unsqueeze(intersection, dim=-1)
            self.state = self.state * intersection + new_state * torch.logical_not(
                intersection)

        reward = self.compute_reward(self.state[:, :2], self.goal, {})
        self.num_steps += 1

        done = torch.full((self.batch_size, 1), False).to(self.device)
        info = {"is_success": self._is_success(self.state[:, :2], self.goal)}

        return (
            {
                "observation": self.state,
                "achieved_goal": self.state[:, :2],
                "desired_goal": self.goal,
            },
            reward,
            done,
            info,
        )
