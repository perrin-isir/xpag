# SGS: SEQUENTIAL GOAL SWITCHING

import bisect
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
        # self.cut_value = -70.
        self.cut_value = -10.
        self.goal_sequence = []
        # self.budget_sequence = []
        # self.episode_budget_sequence = None
        # self.current_budgets = None
        self.wait_sequence = None
        self.current_idxs = None
        self.timesteps = None
        # self.mean_value = None
        self.stats = np.zeros(40).astype('int')
        self.budget = None
        self.global_ts = 0

    def reset(self, obs):
        if self.global_ts > 0:
            # self.mean_value /= self.global_ts
            # self.cut_value = np.mean(self.mean_value) / self.global_ts
            # print(self.mean_value)
            print(self.budget)
        # self.mean_value = []
        self.global_ts = 0
        self.current_idxs = np.zeros(self.num_envs).astype('int')
        # self.episode_budget_sequence = self.budget_sequence.copy()
        # if np.random.random() > 0.:
        #     i = np.random.choice(len(self.episode_budget_sequence))
        #     self.episode_budget_sequence[i] = int(
        #         self.episode_budget_sequence[i] * np.random.random())
        # self.current_budgets = self.episode_budget_sequence[self.current_idxs]
        self.wait_sequence = np.zeros(len(self.goal_sequence)).astype('int')
        if np.random.random() > 0.5:
            i = np.random.choice(len(self.wait_sequence))
            self.wait_sequence[i] = 1
        self.timesteps = np.zeros(self.num_envs).astype('int')
        obs['desired_goal'][:] = self.goal_sequence[self.current_idxs]
        return obs

    def step(self, o, action, new_o, reward, done, info):
        # next_goals = self.goal_sequence[
        #     (self.current_idxs + 1).clip(0, len(self.goal_sequence) - 1)]
        # q_a = self.agent.value(hstack_func(o['observation'], next_goals), action)
        q_a = self.agent.value(hstack_func(o['observation'], o['desired_goal']), action)
        # self.mean_value.append(float(q_a.mean()))
        # self.mean_value.append((float(q_a[0]), self.current_idxs[0]))
        self.global_ts += 1
        self.timesteps += 1
        # values = datatype_convert(
        #     # q_a > self.cut_value,
        #     q_a > self.cut_value,
        #     DataType.NUMPY
        # ).astype('int')
        vals = datatype_convert(q_a, DataType.NUMPY)
        newvalues = np.zeros(self.num_envs).astype('int')
        for k in range(self.num_envs):
            if len(self.budget[k][self.current_idxs[k]]) > 0:
                if vals[k] > self.budget[k][self.current_idxs[k]][0] + self.cut_value:
                    newvalues[k] = 1
                    self.budget[k][self.current_idxs[k]].pop(0)
        # for j in range(len(vals)):
        #     if len(self.budget[j]) > 0:
        #         if vals[j] > self.budget[j][0] + self.cut_value:
        #             newvalues[j] = 1
        #             self.budget[j].pop(0)
        # if info is not None:
        #     info["target"] = ""
        # new_o["desired_goal"] =
        # values = values * (1 - self.wait_sequence[self.current_idxs])
        delta = datatype_convert(info['is_success'], DataType.NUMPY).astype('int')
        # for k in range(self.num_envs):
        #     if delta[k]:
        #         for _ in range(2):
        #             bisect.insort(self.budget[k][self.current_idxs[k]], vals[k])
        # for goalval, success, i in zip(vals, delta, self.current_idxs):
        #     # self.budget[i] += incr
        #     # self.budget[i].append(vals[i])
        #     if success:
        #         bisect.insort(self.budget[i], goalval)
        #         print('ok')
        # for j in range(len(values)):
        #     if values[j] and self.budget[self.current_idxs[j]] > 0:
        #         self.budget[self.current_idxs[j]] -= 1
        #     else:
        #         values[j] = 0
        # delta = np.maximum(delta, (self.timesteps > self.current_budgets).astype('int'))
        # if delta.max():
        #     print('delta')
        # if values.max():
        #     print('value')
        # delta = np.maximum(delta, values)
        delta = np.maximum(delta, newvalues)
        delta_max = delta.max()
        if delta_max:
            self.timesteps = self.timesteps * (1 - delta)
            self.current_idxs += delta
            self.current_idxs = self.current_idxs.clip(0, len(self.goal_sequence) - 1)
            # self.current_budgets = self.episode_budget_sequence[self.current_idxs]
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
                self.budget[k].append([])
        # self.budget_sequence = np.array(bseq)
        # self.budget_sequence = datatype_convert(self.budget_sequence,
        #                                         DataType.NUMPY,
        #                                         self.device)

        # for i in range(len(self.goal_sequence)):
        #     self.goal_sequence[i] = datatype_convert(self.goal_sequence[i],
        #                                              self.datatype,
        #                                              self.device)
        # self.budget_sequence = bseq
