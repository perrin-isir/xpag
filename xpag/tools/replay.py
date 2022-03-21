import os
import numpy as np
import gym
import time
from IPython import display
import matplotlib.pyplot as plt


def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode="rgb_array"))
    plt.title(f"step: {step}")
    plt.axis("off")
    display.clear_output(wait=True)
    display.display(plt.gcf())


def mujoco_notebook_replay(load_dir: str):
    """
    Episode replay for mujoco environments.
    In a notebook, use '%matplotlib inline' for a correct display.
    """
    env_name = str(
        np.loadtxt(os.path.join(load_dir, "episode", "env_name.txt"), dtype="str")
    )
    env_replay = gym.make(env_name)
    qpos = np.load(os.path.join(load_dir, "episode", "qpos.npy"))
    qvel = np.load(os.path.join(load_dir, "episode", "qvel.npy"))
    i = 0
    while True:
        tic = time.time()
        env_replay.set_state(qpos[i], qvel[i])
        env_replay.render(mode="rgb_array")
        show_state(env_replay, i)
        i = (i + 1) % len(qpos)
        toc = time.time()
        elapsed = toc - tic
        # 60 Hz = normal speed
        dt_sleep = max((0, 1.0 / 60.0 - elapsed))
        time.sleep(dt_sleep)
