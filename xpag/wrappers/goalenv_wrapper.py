import gym
from gym.vector import VectorEnv
from gym import spaces
from xpag.tools.utils import get_datatype, datatype_convert, where, hstack
import numpy as np
from typing import Callable, Any


class GoalEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: VectorEnv,
        goal_space: spaces.Space,
        compute_achieved_goal: Callable[[Any], Any],
        compute_reward: Callable[[Any, Any, Any, Any, Any, Any, Any, Any], Any],
        compute_success: Callable[[Any, Any], Any],
        terminate_on_succes: bool = False,
    ):
        super().__init__(env)
        self.datatype = None
        self.last_desired_goal = None
        self.compute_achieved_goal = compute_achieved_goal
        self.compute_reward = compute_reward
        self.compute_success = compute_success
        self.terminate_on_success = terminate_on_succes

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
        obs, info = self.env.reset(**kwargs)
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
        return goalenv_obs, info

    def reset_done(self, *args, **kwargs):
        obs, info = self.env.reset_done(*args, **kwargs)
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
        return goalenv_obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        assert (
            self.last_desired_goal is not None
        ), "reset() or reset_done() must be called before step()"
        goalenv_obs = {
            "observation": observation,
            "achieved_goal": self.compute_achieved_goal(observation),
            "desired_goal": self.last_desired_goal,
        }
        goalenv_reward = datatype_convert(
            self.compute_reward(
                goalenv_obs["achieved_goal"],
                goalenv_obs["desired_goal"],
                action,
                observation,
                reward,
                terminated,
                truncated,
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
        goalenv_terminated = (
            where(
                goalenv_success,
                datatype_convert(np.ones_like(terminated), self.datatype),
                terminated,
            )
            if self.terminate_on_success
            else terminated
        )
        return goalenv_obs, goalenv_reward, goalenv_terminated, truncated, info

    def set_goal(self, goal):
        assert (
            self.datatype is not None
        ), "reset() or reset_done() must be called before set_goal()"
        self.last_desired_goal = datatype_convert(goal, self.datatype)


class CumulRewardWrapper(gym.Wrapper):
    """An environment wrapper that adds the cumulative reward to observations.
    It assumes that the environment is not a goal-based environment, and that
    observations are 1D arrays (with .single_observation_space of type gym.spaces.Box).
    """

    def __init__(
        self,
        env: VectorEnv,
        normalization_factor: float = 1.0,
    ):
        super().__init__(env)
        self.datatype = None
        self.cumulative_reward = None
        self.normalization_factor = normalization_factor
        self.single_observation_space = spaces.Box(
            np.hstack((env.single_observation_space.low, -np.inf)),
            np.hstack((env.single_observation_space.high, np.inf)),
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.datatype is None:
            self.datatype = get_datatype(obs)
        self.cumulative_reward = datatype_convert(
            np.zeros((self.num_envs, 1)), self.datatype
        )
        obs = hstack(obs, self.cumulative_reward)
        return obs, info

    def reset_done(self, done, **kwargs):
        assert (
            self.cumulative_reward is not None
        ), "reset() must be called before reset_done()"
        self.cumulative_reward = where(
            done,
            datatype_convert(np.zeros_like(self.cumulative_reward), self.datatype),
            self.cumulative_reward,
        )
        obs, info = self.env.reset_done(done, **kwargs)
        obs = hstack(obs, self.cumulative_reward)
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        assert (
            self.cumulative_reward is not None
        ), "reset() must be called before step()"
        self.cumulative_reward += reward * self.normalization_factor
        observation = hstack(observation, self.cumulative_reward)
        return observation, reward, terminated, truncated, info
