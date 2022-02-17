# SGS: SEQUENTIAL GOAL SWITCHING

from abc import ABC
from xpag.goalsetters.goalsetter import GoalSetter
from xpag.tools.utils import debug
from xpag.tools.utils import DataType, datatype_convert
import numpy as np


class SGS(GoalSetter, ABC):
    def __init__(self, params=None,
                 num_envs: int = 1,
                 datatype: DataType = DataType.TORCH,
                 device: str = 'cpu'):
        if params is None:
            params = {}
        super().__init__("SGS", params, num_envs, datatype, device)
        self.goal_sequence = []
        self.budget_sequence = []
        self.current_idxs = np.zeros(self.num_envs)
        self.ts = 0

    def reset(self, obs):
        self.current_idxs = np.zeros(self.num_envs)
        self.ts = 0
        # obs["desired_goal"] =
        obs['desired_goal'][:] = self.goal_sequence[0]
        return obs

    def step(self, action, new_o, reward, done, info):
        # if info is not None:
        #     info["target"] = ""
        self.ts += 1
        # new_o["desired_goal"] =
        return action, new_o, reward, done, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass

    def set_sequence(self, gseq, bseq):
        self.goal_sequence = gseq
        for i in range(len(self.goal_sequence)):
            self.goal_sequence[i] = datatype_convert(self.goal_sequence[i],
                                                     self.datatype,
                                                     self.device)
        self.budget_sequence = bseq
