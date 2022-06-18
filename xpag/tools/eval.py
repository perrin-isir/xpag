# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import os
from typing import Union, Dict, Any
import numpy as np
from xpag.agents.agent import Agent
from xpag.goalsetters.goalsetter import GoalSetter
from xpag.tools.utils import DataType, datatype_convert, hstack
from xpag.tools.timing import timing
from xpag.tools.logging import eval_log
from xpag.plotting.plotting import single_episode_plot


class SaveEpisode:
    """To save episodes in Brax or Mujoco environments"""

    def __init__(self, env, env_info):
        self.env = env
        self.env_info = env_info
        self.qpos = []
        self.qrot = []
        self.qvel = []
        self.qang = []

    def update(self):
        if self.env_info["env_type"] == "Brax":
            self.qpos.append(self.env._state.qp.pos.to_py())
            self.qrot.append(self.env._state.qp.rot.to_py())
            self.qvel.append(self.env._state.qp.vel.to_py())
            self.qang.append(self.env._state.qp.ang.to_py())
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
            np.save(
                os.path.join(save_dir, "episode", "qp_pos"),
                [pos[i] for pos in self.qpos],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_rot"),
                [rot[i] for rot in self.qrot],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_vel"),
                [vel[i] for vel in self.qvel],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_ang"),
                [ang[i] for ang in self.qang],
            )
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
        self.qrot = []
        self.qvel = []
        self.qang = []


def single_rollout_eval(
    steps: int,
    eval_env: Any,
    env_info: Dict[str, Any],
    agent: Agent,
    goalsetter: GoalSetter,
    save_dir: Union[str, None] = None,
    plot_projection=None,
    save_episode: bool = False,
    env_datatype: Union[DataType, None] = None,
):
    # Evaluation performed on a single run
    interval_time, _ = timing()
    observation, _ = goalsetter.reset(
        eval_env, *eval_env.reset(return_info=True), eval_mode=True
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
        action = datatype_convert(
            agent.select_action(obs, eval_mode=True), env_datatype
        )
        next_observation, reward, done, info = goalsetter.step(
            eval_env, observation, action, *eval_env.step(action), eval_mode=True
        )
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
