import gym
from gym.vector import VectorEnv
from gym import spaces
import numpy as np


class GoalEnvWrapper(gym.Wrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._achieved_goal_dim = 2
        self._desired_goal_dim = 2
        high_achieved_goal = np.ones(self._achieved_goal_dim)
        low_achieved_goal = -high_achieved_goal
        high_desired_goal = np.ones(self._desired_goal_dim)
        low_desired_goal = -high_desired_goal

        self.single_observation_space = spaces.Dict(
            dict(
                observation=self.env.single_observation_space,
                achieved_goal=spaces.Box(
                    low_achieved_goal, high_achieved_goal, dtype=np.float64
                ),
                desired_goal=spaces.Box(
                    low_desired_goal, high_desired_goal, dtype=np.float64
                ),
            )
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    # def step(self, action: np.ndarray):
    #     _, truncation = self.common_step(action)
    #
    #     reward = self.compute_reward(achieved_g(self.state), self.goal, {}).reshape(
    #         (self.num_envs, 1)
    #     )
    #     is_success = self._is_success(achieved_g(self.state), self.goal).reshape(
    #         (self.num_envs, 1)
    #     )
    #     truncation = truncation * (1 - is_success)
    #     info = {
    #         "is_success": is_success,
    #         "truncation": truncation,
    #     }
    #     self.done = np.maximum(truncation, is_success)
    #
    #     return (
    #         {
    #             "observation": self.state,
    #             "achieved_goal": achieved_g(self.state),
    #             "desired_goal": self.goal,
    #         },
    #         reward,
    #         self.done,
    #         info,
    #     )
