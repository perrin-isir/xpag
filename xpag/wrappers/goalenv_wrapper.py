import gym
from gym.vector import VectorEnv
from gym import spaces
from xpag.tools.utils import get_datatype, datatype_convert, where
import numpy as np
from typing import Callable, Any, Optional


class GoalEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: VectorEnv,
        goal_space: spaces.Space,
        compute_achieved_goal: Callable[[Any, Optional[Any]], Any],
        compute_reward: Callable[[Any, Any, Any, Any, Any, Any, Any], Any],
        compute_success: Callable[[Any, Any], Any],
        done_on_succes: bool = False,
    ):
        super().__init__(env)
        self.datatype = None
        self.last_desired_goal = None
        self.compute_achieved_goal = compute_achieved_goal
        self.compute_reward = compute_reward
        self.compute_success = compute_success
        self.done_on_success = done_on_succes

        self.single_observation_space = spaces.Dict(
            dict(
                observation=self.env.single_observation_space,
                achieved_goal=goal_space,
                desired_goal=goal_space,
            )
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    def reset(self, **kwargs):
        if "return_info" in kwargs and kwargs["return_info"]:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        if self.datatype is None:
            self.datatype = get_datatype(obs)
        self.last_desired_goal = datatype_convert(
            self.observation_space["desired_goal"].sample(), self.datatype
        )
        goalenv_obs = {
            "observation": obs,
            "achieved_goal": self.compute_achieved_goal(obs),
            "desired_goal": self.last_desired_goal,
        }
        if "return_info" in kwargs and kwargs["return_info"]:
            return goalenv_obs, info
        else:
            return goalenv_obs

    def reset_done(self, *args, **kwargs):
        if "return_info" in kwargs and kwargs["return_info"]:
            obs, info = self.env.reset_done(*args, **kwargs)
        else:
            obs = self.env.reset_done(*args, **kwargs)
        if self.datatype is None:
            self.datatype = get_datatype(obs)
        self.last_desired_goal = datatype_convert(
            self.observation_space["desired_goal"].sample(), self.datatype
        )
        goalenv_obs = {
            "observation": obs,
            "achieved_goal": self.compute_achieved_goal(obs),
            "desired_goal": self.last_desired_goal,
        }
        if "return_info" in kwargs and kwargs["return_info"]:
            return goalenv_obs, info
        else:
            return goalenv_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        assert self.last_desired_goal is not None, (
            "reset() or reset_done() must be" "called before step()"
        )
        goalenv_obs = {
            "observation": observation,
            "achieved_goal": self.compute_achieved_goal(observation, reward),
            "desired_goal": self.last_desired_goal,
        }
        goalenv_reward = datatype_convert(
            self.compute_reward(
                goalenv_obs["achieved_goal"],
                goalenv_obs["desired_goal"],
                action,
                observation,
                reward,
                done,
                info,
            ),
            self.datatype,
        )
        goalenv_success = datatype_convert(
            self.compute_success(
                goalenv_obs["achieved_goal"], goalenv_obs["desired_goal"]
            ),
            self.datatype,
        )
        info["is_success"] = goalenv_success
        goalenv_done = where(
            goalenv_success, datatype_convert(np.ones_like(done), self.datatype), done
        )
        return goalenv_obs, goalenv_reward, goalenv_done, info

    def set_goal(self, goal):
        assert self.datatype is not None, (
            "reset() or reset_done() must be" "called before set_goal()"
        )
        self.last_desired_goal = datatype_convert(goal, self.datatype)
