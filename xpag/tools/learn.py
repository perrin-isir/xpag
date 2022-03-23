import os
import numpy as np
from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
from xpag.buffers import Buffer
from xpag.agents.agent import Agent
from xpag.goalsetters.goalsetter import GoalSetter
from typing import Dict, Any, Union


def learn(
    env,
    eval_env,
    env_info: Dict[str, Any],
    agent: Agent,
    buffer: Buffer,
    goalsetter: GoalSetter,
    batch_size: int = 256,
    gd_steps_per_step: int = 1,
    start_training_after_x_steps: int = 0,
    max_steps: int = 1_000_000_000,
    evaluate_every_x_steps: int = np.inf,
    save_agent_every_x_steps: int = np.inf,
    save_dir: Union[None, str] = None,
    save_episode: bool = False,
    plot_projection=None,
):
    eval_log_reset()
    timing_reset()
    observation = goalsetter.reset(env.reset())

    for i in range(max_steps // env_info["num_envs"]):
        if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
            single_rollout_eval(
                i * env_info["num_envs"],
                eval_env,
                env_info,
                agent,
                save_dir=save_dir,
                plot_projection=plot_projection,
                save_episode=save_episode,
            )

        if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(save_dir, "agent"))

        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            action = agent.select_action(
                observation
                if not env_info["is_goalenv"]
                else hstack(observation["observation"], observation["desired_goal"]),
                deterministic=False,
            )
            for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
                _ = agent.train_on_batch(buffer.sample(batch_size))

        next_observation, reward, done, info = goalsetter.step(*env.step(action))

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
        buffer.insert(step)
        observation = next_observation

        if done.max():
            # use store_done() if the buffer is an episodic buffer
            if hasattr(buffer, "store_done"):
                buffer.store_done()
            observation = goalsetter.reset_done(env.reset_done())
