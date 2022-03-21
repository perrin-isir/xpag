import os
import numpy as np
import gym
import time
from IPython import display

# import imageio
# import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def show_img(img, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(img)
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
        display.clear_output(wait=True)
        img = Image.fromarray(
            env_replay.render(mode="rgb_array", width=320, height=240)
        )
        ImageDraw.Draw(img).text(
            (0, 0), f"step: {i}", (255, 255, 255)  # Coordinates  # Text  # Color
        )
        display.display(img)
        i = (i + 1) % len(qpos)
        toc = time.time()
        elapsed = toc - tic
        dt_sleep = max((0, env_replay.model.opt.timestep - elapsed))
        time.sleep(dt_sleep)

    #
    # if False:
    #     i = 0
    #     while True:
    #         # tic = time.time()
    #         env_replay.set_state(qpos[i], qvel[i])
    #         show_img(env_replay.render(mode="rgb_array"), i)
    #         i = (i + 1) % len(qpos)
    #         # toc = time.time()
    #         # elapsed = toc - tic
    #         # 60 Hz = normal speed
    #         # dt_sleep = max((0, 1.0 / 60.0 - elapsed))
    #         # time.sleep(dt_sleep)
    # else:
    #     gif_writer = imageio.get_writer(
    #         os.path.join(load_dir, 'replay.gif'), mode='I', duration=1. / 60)
    #     img_list = []
    #     for i in range(len(qpos)):
    #         env_replay.set_state(qpos[i], qvel[i])
    #         img_list.append(env_replay.render(mode="rgb_array"))
    #         if not i % 100:
    #             print(i)
    #     i = 0
    #     while True:
    #         show_img(img_list[i], i)
    #         i = (i + 1) % len(qpos)
    #         # gif_writer.append_data(env_replay.render(mode="rgb_array"))
