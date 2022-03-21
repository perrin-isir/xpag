import os
import numpy as np
import gym

# import time
from IPython import display
from PIL import Image, ImageDraw

# import matplotlib.pyplot as plt
import ipywidgets

# from ipywidgets import interact, Layout


# def show_img(img, step=0, info=""):
#     plt.figure(3)
#     plt.clf()
#     plt.imshow(img)
#     plt.title(f"step: {step}")
#     plt.axis("off")
#     display.clear_output(wait=True)
#     display.display(plt.gcf())


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
    # img_list = []
    # for i in range(len(qpos)):
    #     tic = time.time()
    #     env_replay.set_state(qpos[i], qvel[i])
    #     display.clear_output(wait=True)
    #     img = Image.fromarray(
    #         env_replay.render(mode="rgb_array", width=320, height=240)
    #     )
    #     ImageDraw.Draw(img).text(
    #         (0, 0), f"step: {i}", (255, 255, 255)  # Coordinates  # Text  # Color
    #     )
    #     img_list.append(img)
    #     display.display(img)
    #     toc = time.time()
    #     elapsed = toc - tic
    #     dt_sleep = max((0, env_replay.model.opt.timestep - elapsed))
    #     time.sleep(dt_sleep)

    img_dict = {}

    play = ipywidgets.Play(
        # interval=1000,
        value=0,
        min=0,
        max=len(qpos) - 1,
        step=1,
        description="Press play",
        disabled=False,
    )

    def display_sequence(slider_):
        def _show(step):
            if step in img_dict:
                return img_dict[step]
            else:
                env_replay.set_state(qpos[step], qvel[step])
                img_ = Image.fromarray(env_replay.render(mode="rgb_array"))
                ImageDraw.Draw(img_).text((0, 0), f"step: {step}", (255, 255, 255))
                img_dict[step] = img_
                return img_dict[step]

        return ipywidgets.interact(_show, step=slider_)

    slider = ipywidgets.IntSlider(
        min=0,
        max=len(qpos) - 1,
        step=1,
        value=0,
        readout=True,
        layout=ipywidgets.Layout(width="400px"),
    )
    ipywidgets.jslink((play, "value"), (slider, "value"))
    display.display(ipywidgets.HBox([play]))
    # display_sequence(img_list, slider)
    display_sequence(slider)
    # return img_list
    # i = 0
    # while True:
    #     tic = time.time()
    #     display.clear_output(wait=True)
    #     display.display(img_list[i])
    #     i = (i + 1) % len(qpos)
    #     toc = time.time()
    #     elapsed = toc - tic
    #     dt_sleep = max((0, env_replay.model.opt.timestep - elapsed))
    #     time.sleep(dt_sleep)
