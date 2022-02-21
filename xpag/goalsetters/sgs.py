# SGS: SEQUENTIAL GOAL SWITCHING

import bisect
from collections import deque
from abc import ABC
from xpag.goalsetters.goalsetter import GoalSetter
from xpag.tools.utils import debug, DataType, datatype_convert, hstack_func
import numpy as np


class SGS(GoalSetter, ABC):
    def __init__(self, params,
                 num_envs: int = 1,
                 datatype: DataType = DataType.TORCH,
                 device: str = 'cpu'):
        if params is None:
            params = {}
        super().__init__("SGS", params, num_envs, datatype, device)
        self.agent = self.params['agent']
        self.cut_steps = 5
        self.goal_sequence = []
        self.budget_sequence = []
        self.current_idxs = None
        self.timesteps = None
        self.q_a = None
        self.budget = None
        self.global_ts = 0

    def reset(self, obs):
        self.q_a = deque(maxlen=5)
        self.global_ts = 0
        self.current_idxs = np.zeros(self.num_envs).astype('int')
        self.timesteps = np.zeros(self.num_envs).astype('int')
        obs['desired_goal'][:] = self.goal_sequence[self.current_idxs]
        return obs

    def step(self, o, action, new_o, reward, done, info):
        # next_goals = self.goal_sequence[
        #     (self.current_idxs + 1).clip(0, len(self.goal_sequence) - 1)]
        # q_a = self.agent.value(hstack_func(o['observation'], next_goals), action)
        q_a = self.agent.value(hstack_func(o['observation'], o['desired_goal']), action)
        self.global_ts += 1
        self.timesteps += 1
        self.q_a.append(datatype_convert(q_a, DataType.NUMPY))
        newvalues = np.zeros(self.num_envs).astype('int')
        for k in range(self.num_envs):
            if len(self.budget[k][self.current_idxs[k]]) > 0:
                if self.q_a[-1][k] > self.budget[k][self.current_idxs[k]][0] and \
                        self.budget[k][self.current_idxs[k]][1] > 0:
                    newvalues[k] = 1
                    self.budget[k][self.current_idxs[k]] = \
                        self.budget[k][self.current_idxs[k]][0], \
                        self.budget[k][self.current_idxs[k]][1] - 1

        delta = datatype_convert(info['is_success'], DataType.NUMPY).astype('int')
        for k in range(self.num_envs):
            if delta[k]:
                _, n = self.budget[k][self.current_idxs[k]]
                self.budget[k][self.current_idxs[k]] = (self.q_a[0][k], min(n + 2, 20))

        delta = np.maximum(delta, newvalues)
        delta_max = delta.max()
        if delta_max:
            self.timesteps = self.timesteps * (1 - delta)
            self.current_idxs += delta
            self.current_idxs = self.current_idxs.clip(0, len(self.goal_sequence) - 1)
            new_o['desired_goal'][:] = self.goal_sequence[self.current_idxs]
        return o, action, new_o, reward, done, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass

    def set_sequence(self, gseq, bseq):
        self.goal_sequence = np.array(gseq)
        self.goal_sequence = datatype_convert(self.goal_sequence,
                                              self.datatype,
                                              self.device)
        # self.budget = np.zeros(len(self.goal_sequence)).astype('int')
        self.budget = []
        for k in range(self.num_envs):
            self.budget.append([])
            for i in range(len(self.goal_sequence)):
                self.budget[k].append((0., 0))
        self.budget_sequence = np.array(bseq)
        self.budget_sequence = datatype_convert(self.budget_sequence,
                                                DataType.NUMPY,
                                                self.device)
