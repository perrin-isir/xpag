# SGS: SEQUENTIAL GOAL SWITCHING

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
        self.cut_value = -20.
        self.goal_sequence = []
        self.budget_sequence = []
        self.episode_budget_sequence = None
        self.current_idxs = None
        self.current_budgets = None
        self.timesteps = None
        self.mean_value = None
        self.global_ts = 0

    def reset(self, obs):
        if self.global_ts > 0:
            # self.mean_value /= self.global_ts
            # self.cut_value = np.mean(self.mean_value) / self.global_ts
            print(self.mean_value)
        self.mean_value = []
        self.global_ts = 0
        self.current_idxs = np.zeros(self.num_envs).astype('int')
        self.episode_budget_sequence = self.budget_sequence.copy()
        if np.random.random() > 0.:
            i = np.random.choice(len(self.episode_budget_sequence))
            self.episode_budget_sequence[i] = int(
                self.episode_budget_sequence[i] * np.random.random())
        self.current_budgets = self.episode_budget_sequence[self.current_idxs]
        self.timesteps = np.zeros(self.num_envs).astype('int')
        obs['desired_goal'][:] = self.goal_sequence[self.current_idxs]
        return obs

    def step(self, o, action, new_o, reward, done, info):
        next_goals = self.goal_sequence[
            (self.current_idxs + 1).clip(0, len(self.goal_sequence) - 1)]
        q_a = self.agent.value(hstack_func(o['observation'], next_goals), action)
        # q_a = self.agent.value(hstack_func(o['observation'], o['desired_goal']), action)
        self.mean_value.append(float(q_a.mean()))
        self.global_ts += 1
        values = datatype_convert(
            q_a > self.cut_value,
            DataType.NUMPY
        ).astype('int')
        # if info is not None:
        #     info["target"] = ""
        self.timesteps += 1
        # new_o["desired_goal"] =
        delta = datatype_convert(info['is_success'], DataType.NUMPY).astype('int')
        delta = np.maximum(delta, (self.timesteps > self.current_budgets).astype('int'))
        delta = np.maximum(delta, values)
        delta_max = delta.max()
        if delta_max:
            self.timesteps = self.timesteps * (1 - delta)
            self.current_idxs += delta
            self.current_idxs = self.current_idxs.clip(0, len(self.goal_sequence) - 1)
            self.current_budgets = self.episode_budget_sequence[self.current_idxs]
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
        self.budget_sequence = np.array(bseq)
        self.budget_sequence = datatype_convert(self.budget_sequence,
                                                DataType.NUMPY,
                                                self.device)

        # for i in range(len(self.goal_sequence)):
        #     self.goal_sequence[i] = datatype_convert(self.goal_sequence[i],
        #                                              self.datatype,
        #                                              self.device)
        # self.budget_sequence = bseq
