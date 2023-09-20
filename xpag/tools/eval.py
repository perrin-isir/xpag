# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

import os
from typing import Union, Dict, Any, Optional
import numpy as np
from xpag.agents.agent import Agent
from xpag.setters.setter import Setter
from xpag.tools.utils import DataType, datatype_convert, hstack, logical_or
from xpag.tools.timing import timing
from xpag.tools.logging import eval_log
from xpag.plotting.plotting import single_episode_plot
import joblib


class SaveEpisode:
    """To save episodes in Brax or Mujoco environments"""

    def __init__(self, env, env_info):
        self.env = env
        self.env_info = env_info
        self.qpos = []
        self.qvel = []
        self.states = []

    def update(self):
        if self.env_info["env_type"] == "Brax":
            self.states.append(self.env.unwrapped._state)
        elif self.env_info["env_type"] == "Mujoco":
            posvel = np.split(
                np.array(self.env.call("state_vector")),
                [self.env.call("init_qpos")[0].shape[-1]],
                axis=1,
            )
            self.qpos.append(posvel[0])
            self.qvel.append(posvel[1])
        else:
            pass

    def save(self, i: int, save_dir: str):
        os.makedirs(os.path.join(save_dir, "episode"), exist_ok=True)
        if self.env_info["env_type"] == "Brax":
            with open(os.path.join(save_dir, "episode", "env_name.txt"), "w") as f:
                print(self.env_info["name"], file=f)
            with open(
                os.path.join(save_dir, "episode", "episode_length.txt"), "w"
            ) as f:
                print(len(self.states), file=f)
            with open(os.path.join(save_dir, "episode", "states.joblib"), "wb") as f:
                joblib.dump(self.states, f)
        elif self.env_info["env_type"] == "Mujoco":
            with open(os.path.join(save_dir, "episode", "env_name.txt"), "w") as f:
                print(self.env_info["name"], file=f)
            np.save(
                os.path.join(save_dir, "episode", "qpos"), [pos[i] for pos in self.qpos]
            )
            np.save(
                os.path.join(save_dir, "episode", "qvel"), [vel[i] for vel in self.qvel]
            )

        self.qpos = []
        self.qvel = []
        self.states = []


def single_rollout_eval(
    steps: int,
    eval_env: Any,
    env_info: Dict[str, Any],
    agent: Agent,
    setter: Setter,
    save_dir: Union[str, None] = None,
    plot_projection=None,
    save_episode: bool = False,
    env_datatype: Optional[DataType] = None,
    seed: Optional[int] = None,
) -> float:
    """Evaluation performed on a single run"""
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    interval_time, _ = timing()
    observation, _ = setter.reset(
        eval_env,
        *eval_env.reset(seed=master_rng.randint(1e9)),
        eval_mode=True,
    )
    if save_episode and save_dir is not None:
        save_ep = SaveEpisode(eval_env, env_info)
        save_ep.update()
    done = np.array(False)
    cumulated_reward = 0.0
    step_list = []
    while not done.max():
        obs = (
            observation
            if not env_info["is_goalenv"]
            else hstack(observation["observation"], observation["desired_goal"])
        )
        action = agent.select_action(obs, eval_mode=True)
        action_info = {}
        if isinstance(action, tuple):
            action_info = action[1]
            action = action[0]
        action = datatype_convert(action, env_datatype)
        (
            observation,
            action,
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = setter.step(
            eval_env,
            observation,
            action,
            action_info,
            *eval_env.step(action),
            eval_mode=True,
        )
        done = logical_or(terminated, truncated)
        if save_episode and save_dir is not None:
            save_ep.update()
        cumulated_reward += reward.mean()
        step_list.append(
            {"observation": observation, "next_observation": next_observation}
        )
        observation = next_observation
    eval_log(
        steps,
        interval_time,
        cumulated_reward,
        None if not env_info["is_goalenv"] else info["is_success"].mean(),
        env_info,
        agent,
        save_dir,
    )
    if plot_projection is not None and save_dir is not None:
        os.makedirs(os.path.join(os.path.expanduser(save_dir), "plots"), exist_ok=True)
        single_episode_plot(
            os.path.join(
                os.path.expanduser(save_dir),
                "plots",
                f"{steps:12}.png".replace(" ", "0"),
            ),
            step_list,
            projection_function=plot_projection,
            plot_env_function=None if not hasattr(eval_env, "plot") else eval_env.plot,
        )
    if save_episode and save_dir is not None:
        save_ep.save(0, os.path.expanduser(save_dir))
    timing()
    return cumulated_reward
