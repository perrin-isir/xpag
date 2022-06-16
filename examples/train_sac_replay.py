import os

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


def main():

    use_brax_rp = True
    use_sac = True

    num_envs = 50  # the number of rollouts in parallel during training
    env, eval_env, env_info = brax_vec_env("walker2d", num_envs)

    batch_size = 256
    if use_brax_rp:
        buffer_size = batch_size + 1
    else:
        buffer_size = 1_000_000

    agent_class = SAC if use_sac else TD3
    agent = agent_class(
        env_info["observation_dim"]
        if not env_info["is_goalenv"]
        else env_info["observation_dim"] + env_info["desired_goal_dim"],
        env_info["action_dim"],
        {"backend": "gpu"},
    )

    goalsetter = DefaultGoalSetter()
    if use_brax_rp:
        dummy_step = {
            "observation": jnp.zeros(env_info["observation_dim"]),
            "action": jnp.zeros(env_info["action_dim"]),
            "reward": jnp.zeros(1),
            "truncation": jnp.zeros(1),
            "done": jnp.zeros(1),
            "next_observation": jnp.zeros(env_info["observation_dim"]),
        }
        buffer = BraxUniformSamplingQueueToXpag(buffer_size=buffer_size, sampler=None)
        buffer.init_rp_buffer(dummy_step)
    else:
        sampler = (
            DefaultEpisodicSampler()
            if not env_info["is_goalenv"]
            else HER(env.compute_reward)
        )
        buffer = DefaultEpisodicBuffer(
            max_episode_steps=env_info["max_episode_steps"],
            buffer_size=buffer_size,
            sampler=sampler,
        )

    gd_steps_per_step = 1
    start_training_after_x_steps = 256
    max_steps = 50_000
    evaluate_every_x_steps = 5_000
    save_agent_every_x_steps = 100_000
    save_dir = os.path.join(os.path.expanduser("~"), "results", "xpag", "train_brax")
    save_episode = True
    plot_projection = None
    import cProfile
    from pstats import Stats

    # pr = cProfile.Profile()
    # pr.enable()

    learn(
        env,
        eval_env,
        env_info,
        agent,
        buffer,
        goalsetter,
        batch_size,
        gd_steps_per_step,
        start_training_after_x_steps,
        max_steps,
        evaluate_every_x_steps,
        save_agent_every_x_steps,
        save_dir,
        save_episode,
        plot_projection,
    )
    # pr.disable()
    # stats = Stats(pr)
    # stats = stats.sort_stats("tottime")
    # algo_name = "sac" if use_sac else "td3"
    # buffer_name = "brax_rp" if use_brax_rp else "xpag_rp"
    # stats.dump_stats(
    #     f"{os.getcwd()}/cProfile/full_train_{algo_name}_{buffer_name}.data"
    # )


if __name__ == "__main__":

    main()
