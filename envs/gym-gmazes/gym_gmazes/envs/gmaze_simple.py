import gym
from gym import utils, spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc


def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[0], b[0], c[0], d[0]
    y1, y2, y3, y4 = a[1], b[1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    if denom == 0:
        return False
    else:
        t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
        if t < 0 or t > 1:
            return False
        else:
            t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
            if t < 0 or t > 1:
                return False
            else:
                return True


def default_reward_fun(old_obs, act_, repeat, obs):
    # return obs[0]
    if obs[0] > 0.5 and obs[1] > 0.5:
        return 1
    return 0


class GMazeSimple(gym.Env, utils.EzPickle):
    def __init__(
        self,
        frame_skip=2,
        reward_function=default_reward_fun,
        walls=[
            ([0.5, -0.5], [0.5, 1.0]),
            ([-0.5, -0.5], [-0.5, 1.0]),
            ([0.0, -1.0], [0.0, 0.5]),
        ],
    ):
        utils.EzPickle.__init__(**locals())
        self.reward_function = (
            reward_function  # the reward function is not defined by the environment
        )
        self.frame_skip = (
            frame_skip  # a call to step() repeats the action frame_skip times
        )

        self.init_qpos = np.array([-1.0, 0.0])  # the initial position
        self.init_qvel = np.array([])  # velocities are not used in SimpleMazeEnv
        self.state = np.copy(self.init_qpos)  # the current state (no velocity)
        self.walls = walls

        self._obs_dim = 2
        self._action_dim = 2
        self._max_episode_steps = 50
        self.num_steps = 0
        high = 1.0 * np.ones(self._action_dim)
        low = -high
        self.action_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )  # action_space.shape = (2,)
        high = 1.0 * np.ones(self._obs_dim)
        low = -high
        self.observation_space = spaces.Box(
            low, high, dtype=np.float32
        )  # observation_space.shape = (2,)

    def step(self, action, cell_repertory_on=True):
        # add the vector 'action' to the state frame_skip times, with -1 & +1 boundaries
        old_observation = np.copy(self.state)
        for k in range(self.frame_skip):
            new_state = (self.state + action / 10.0).clip(-1.0, 1.0)
            bool_val = True
            for (w1, w2) in self.walls:
                if intersect(self.state, new_state, w1, w2) is True:
                    bool_val = False
            if bool_val is True:
                self.state = new_state
        observation = np.copy(self.state)  # the new state
        reward = self.reward_function(
            old_observation, action, self.frame_skip, observation
        )
        self.num_steps += 1
        done = False
        if self.num_steps == self._max_episode_steps:
            done = True
        info = None  # no info
        return observation, reward, done, info

    def state_vector(self):
        return np.copy(self.state)  # the current state

    def reset_model(self):
        self.state = np.copy(self.init_qpos)  # reset state to initial value

    def reset(self):
        self.reset_model()  # reset state to initial value
        self.num_steps = 0
        return np.copy(self.state)

    def set_state(self, qpos, qvel=None):
        self.state = np.copy(qpos)  # sets state

    def plot(self):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        fig, ax = plt.subplots()
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

    def plotpaths(self, paths, plot_ax=None):
        if plot_ax is None:
            _, ax = plt.subplots()
        else:
            ax = plot_ax
        lines = []
        rgbs = []
        lines.append(([-1, -1], [-1, 1]))
        rgbs.append((0, 0, 0, 1))
        lines.append(([-1, 1], [1, 1]))
        rgbs.append((0, 0, 0, 1))
        lines.append(([1, 1], [1, -1]))
        rgbs.append((0, 0, 0, 1))
        lines.append(([1, -1], [-1, -1]))
        rgbs.append((0, 0, 0, 1))
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        lenpaths = len(paths)
        for i, path in enumerate(paths):
            lines = []
            rgbs = []
            for p in path:
                lines.append((p["obs"], p["obs_next"]))
                rgbs.append((1.0 - i / lenpaths, 0.2, 0.2, 1))
            ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.savefig(filename, dpi=200)
        # plt.close()
