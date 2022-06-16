import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from xpag.agents import SAC
from xpag.wrappers import brax_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.agents import TD3

from xpag.tools import learn, brax_notebook_replay

from examples.replay_buffer_wrapper import BraxUniformSamplingQueueToXpag
import jax
import jax.numpy as jnp
import timeit
import numpy as np
from jax import device_put


def benchmark_fullupdate(use_sac=False, use_brax_rp=False, only_rp=False):
    n_repetition = 500
    num_envs = 50
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
    step_batch = {
        "observation": jnp.zeros((num_envs, env_info["observation_dim"])),
        "action": jnp.zeros((num_envs, env_info["action_dim"])),
        "reward": jnp.zeros((num_envs, 1)),
        "truncation": jnp.zeros((num_envs, 1)),
        "done": jnp.zeros((num_envs, 1)),
        "next_observation": jnp.zeros((num_envs, env_info["observation_dim"])),
    }

    if use_brax_rp:
        dummy_step = {k: v[0] for k, v in step_batch.items()}
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

    while buffer.buffer_size < batch_size:
        buffer.insert(step_batch)
    if only_rp:

        def to_benchmark():
            buffer.insert(step_batch)
            batch = buffer.sample(batch_size)
            for el in batch.values():
                if isinstance(el, jax.numpy.ndarray):
                    el.block_until_ready()

    else:

        def to_benchmark():
            buffer.insert(step_batch)
            batch = buffer.sample(batch_size)
            agent.train_on_batch(batch)
            action = agent.select_action(batch["observation"])
            if isinstance(action, jax.numpy.ndarray):
                action.block_until_ready()

    # warm_up
    to_benchmark()

    algo_name = "sac" if use_sac else "td3"
    buffer_name = "brax_rp" if use_brax_rp else "xpag_rp"
    # with jax.profiler.trace(f"tmp/train_and_rp_{algo_name}_{buffer_name}"):
    elapsed = timeit.timeit(to_benchmark, number=n_repetition)
    if only_rp:
        print(f"onlyBuffer_{buffer_name} time : ", n_repetition / elapsed, "step/sec")
    else:
        print(f"{algo_name}_{buffer_name} time : ", n_repetition / elapsed, "step/sec")


def benchmark_only_update(use_sac=True, use_gpu=True):
    n_repetition = 500
    num_envs = 50
    env, eval_env, env_info = brax_vec_env("walker2d", num_envs)
    batch_size = 256

    agent_class = SAC if use_sac else TD3
    agent = agent_class(
        env_info["observation_dim"]
        if not env_info["is_goalenv"]
        else env_info["observation_dim"] + env_info["desired_goal_dim"],
        env_info["action_dim"],
        # {"backend": "gpu" if use_gpu else "cpu"},
        {"backend": "gpu"},
    )
    module = jnp if use_gpu else np
    batch = {
        "observation": module.zeros((batch_size, env_info["observation_dim"])),
        "action": module.zeros((batch_size, env_info["action_dim"])),
        "reward": module.zeros((batch_size, 1)),
        "truncation": module.zeros((batch_size, 1)),
        "done": module.zeros((batch_size, 1)),
        "next_observation": module.zeros((batch_size, env_info["observation_dim"])),
    }
    if use_gpu:
        for k, el in batch.items():
            assert el.device() == jax.devices("gpu")[0]
            # commit data to the gpu device
            batch[k] = device_put(el, jax.devices("gpu")[0])

    def to_benchmark():
        agent.train_on_batch(batch)
        action = agent.select_action(batch["observation"])
        if isinstance(action, jax.numpy.ndarray):
            action.block_until_ready()

    # warm_up
    to_benchmark()

    algo_name = "sac" if use_sac else "td3"
    device_name = "gpu" if use_gpu else "cpu"
    elapsed = timeit.timeit(to_benchmark, number=n_repetition)
    print(
        f"only_update_{algo_name}_{device_name} time : ",
        n_repetition / elapsed,
        "step/sec",
    )


def device_transfert_benchmark(n_repetition = 5000):
    env_info = {"observation_dim": 20, "action_dim": 6}
    batch_size = 256

    step_batch = {
            "observation": np.zeros((batch_size, env_info["observation_dim"])),
            "action": np.zeros((batch_size, env_info["action_dim"])),
            "reward": np.zeros((batch_size, 1)),
            "truncation": np.zeros((batch_size, 1)),
            # "done": np.zeros((batch_size, 1)),
            "next_observation": np.zeros((batch_size, env_info["observation_dim"])),
        }
    
    def to_benchmark():
        
        # step_batch = jax.tree_util.tree_map(
        #     lambda x: device_put(x, jax.devices("cpu")[0]), step_batch
        # )
        # jax.tree_util.tree_map(
        #     lambda x: device_put(x, jax.devices("gpu")[0]), step_batch
        # )
        jax.tree_util.tree_map(
            lambda x:jnp.array(x), step_batch
        )
    # # warm up : 
    # timeit.timeit(to_benchmark, number=n_repetition)
    # exp : 
    elapsed = timeit.timeit(to_benchmark, number=n_repetition)
    print(
        f"data device transfert time : ",
        n_repetition / elapsed,
        "step/sec",
        f'\n total : {elapsed} for {n_repetition} steps'
    )


def profile_func(function, profile_name, **kwargs):
    import cProfile
    from pstats import Stats

    pr = cProfile.Profile()
    pr.enable()

    function(**kwargs)
    pr.disable()
    stats = Stats(pr)
    stats = stats.sort_stats("tottime")
    stats.dump_stats(f"{os.getcwd()}/cProfile/{profile_name}")
    # stats.print_stats(100)


if __name__ == "__main__":
    # device_transfert_benchmark(5000)
    # profile_func(benchmark_fullupdate, "sac_brax_rp", use_sac=True, use_brax_rp=True)
    # profile_func(benchmark_fullupdate, "sac_xpag_rp", use_sac=True, use_brax_rp=False)
    # profile_func(benchmark_fullupdate, "td3_brax_rp", use_sac=False, use_brax_rp=True)
    # profile_func(benchmark_fullupdate, "td3_xpag_rp", use_sac=False, use_brax_rp=False)

    benchmark_only_update(use_sac=True, use_gpu=True)
    benchmark_only_update(use_sac=True, use_gpu=False)
    benchmark_only_update(use_sac=False, use_gpu=True)
    benchmark_only_update(use_sac=False, use_gpu=False)
