import os
import numpy as np
from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import get_datatype, datatype_convert, hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
from xpag.buffers import Buffer
from xpag.agents.agent import Agent
from xpag.setters.setter import Setter
from typing import Dict, Any, Union, List


def learn(
    env,
    eval_env,
    env_info: Dict[str, Any],
    agent: Agent,
    buffer: Buffer,
    setter: Setter,
    batch_size: int = 256,
    gd_steps_per_step: int = 1,
    start_training_after_x_steps: int = 0,
    max_steps: int = 1_000_000_000,
    evaluate_every_x_steps: int = np.inf,
    save_every_x_steps: int = np.inf,
    save_dir: Union[None, str] = None,
    save_episode: bool = False,
    plot_projection=None,
    custom_eval_function=None,
    additional_step_keys: Union[List[str], None] = None,
):
    eval_log_reset()
    timing_reset()
    reset_obs, reset_info = env.reset(return_info=True)
    env_datatype = get_datatype(
        reset_obs if not env_info["is_goalenv"] else reset_obs["observation"]
    )
    observation, _ = setter.reset(env, reset_obs, reset_info)

    episodic_buffer = True if hasattr(buffer, "store_done") else False

    if custom_eval_function is None:
        rollout_eval = single_rollout_eval
    else:
        rollout_eval = custom_eval_function

    for i in range(max_steps // env_info["num_envs"]):
        if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
            rollout_eval(
                i * env_info["num_envs"],
                eval_env,
                env_info,
                agent,
                setter,
                save_dir=save_dir,
                plot_projection=plot_projection,
                save_episode=save_episode,
                env_datatype=env_datatype,
            )

        if not i % max(save_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(os.path.expanduser(save_dir), "agent"))
                setter.save(os.path.join(os.path.expanduser(save_dir), "setter"))

        action_info = {}
        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            action = agent.select_action(
                observation
                if not env_info["is_goalenv"]
                else hstack(observation["observation"], observation["desired_goal"]),
                eval_mode=False,
            )
            if isinstance(action, tuple):
                action_info = action[1]
                action = action[0]
            if i > 0:
                for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
                    _ = agent.train_on_batch(buffer.sample(batch_size))

        action = datatype_convert(action, env_datatype)

        next_observation, reward, done, info = setter.step(
            env, observation, action, action_info, *env.step(action)
        )

        step = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "truncation": info["truncation"],
            "done": done,
            "next_observation": next_observation,
        }
        if env_info["is_goalenv"]:
            step["is_success"] = info["is_success"]
        if additional_step_keys is not None:
            for a_s_key in additional_step_keys:
                if a_s_key in info:
                    step[a_s_key] = info[a_s_key]
        buffer.insert(step)
        observation = next_observation

        # use store_done() if the buffer is an episodic buffer
        if episodic_buffer:
            buffer.store_done(done)

        if done.max():
            observation, _ = setter.reset_done(
                env, *env.reset_done(done, return_info=True)
            )
