import os
import timeit

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from xpag.agents import SAC
from xpag.wrappers import brax_vec_env, gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import TD3

from xpag.tools import learn, brax_notebook_replay

from examples.replay_buffer_wrapper import BraxUniformSamplingQueueToXpag
import jax.numpy as jnp


def benchmark_env(use_sac=True):

    num_envs = 50  # the number of rollouts in parallel during training
    env, eval_env, env_info = brax_vec_env("walker2d", num_envs)
    n_repetition = 1000
    agent = TD3(
        env_info["observation_dim"]
        if not env_info["is_goalenv"]
        else env_info["observation_dim"] + env_info["desired_goal_dim"],
        env_info["action_dim"],
        {"backend": "gpu"},
    )
    obs = env.reset()
    carry = {"last_obs": obs}

    def to_benchmark():
        obs = carry["last_obs"]
        if use_sac:
            action = agent.select_action(obs)
        else:
            action = jnp.ones((num_envs, env_info["action_dim"]))
        next_observation, _, done, _ = env.step(action)
        # if done.max():
        #     next_observation, _ = env.reset_done(done, return_info=True)
        carry["last_obs"] = next_observation
        next_observation.block_until_ready()

    to_benchmark()

    elapsed = timeit.timeit(to_benchmark, number=n_repetition)
    algo = "SAC" if use_sac else "no_actor"
    print(f"timestep pers seconds {algo} : ", (n_repetition * num_envs) / elapsed)


if __name__ == "__main__":
    benchmark_env(False)
    benchmark_env(True)
