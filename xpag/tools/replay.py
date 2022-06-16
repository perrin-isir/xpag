import os
import numpy as np
import gym
from IPython import display
from PIL import Image, ImageDraw
import ipywidgets
from typing import Callable
from IPython.display import HTML
from brax import envs
from brax.physics.base import QP
from brax.io import html


class DownloadButton(ipywidgets.Button):
    """Download button with dynamic content

    The content is generated using a callback when the button is clicked.
    """

    def __init__(self, contents: Callable[[], str], **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.contents = contents
        self.on_click(self.__on_click)

    def __on_click(self, b):
        filepath = self.contents(self)
        self.description = "99%"
        self.description = "Generate gif"
        print(f"gif saved to: {filepath}")


def mujoco_notebook_replay(load_dir: str):
    """
    Episode replay for mujoco environments.
    """
    env_name = str(
        np.loadtxt(os.path.join(load_dir, "episode", "env_name.txt"), dtype="str")
    )
    env_replay = gym.make(env_name)
    env_replay.reset()
    qpos = np.load(os.path.join(load_dir, "episode", "qpos.npy"))
    qvel = np.load(os.path.join(load_dir, "episode", "qvel.npy"))

    img_dict = {}

    play = ipywidgets.Play(
        value=0,
        min=0,
        max=len(qpos) - 1,
        step=1,
        description="Press play",
        disabled=False,
    )

    def compute_image(step):
        env_replay.set_state(qpos[step], qvel[step])
        img_ = Image.fromarray(env_replay.render(mode="rgb_array"))
        ImageDraw.Draw(img_).text((0, 0), f"step: {step}", (255, 255, 255))
        return img_

    def display_sequence(slider_):
        def _show(step):
            if step in img_dict:
                return img_dict[step]
            else:
                img_dict[step] = compute_image(step)
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
    display_sequence(slider)

    def create_gif(button):
        latest_percent = 0
        for step in range(len(qpos)):
            new_percent = min(int(step / len(qpos) * 100.0), 95)
            if new_percent > latest_percent:
                latest_percent = new_percent
                button.description = f"{latest_percent}%"
            if step not in img_dict:
                img_dict[step] = compute_image(step)
        img_dict[0].save(
            os.path.join(load_dir, "episode", "episode.gif"),
            format="gif",
            append_images=[img_dict[k] for k in range(1, len(qpos))],
            save_all=True,
            duration=env_replay.model.opt.timestep * 1000,
            loop=0,
        )
        button.description = "98%"
        return os.path.join(load_dir, "episode", "episode.gif")

    display.display(DownloadButton(contents=create_gif, description="Generate gif"))


def brax_notebook_replay(load_dir: str):
    """
    Episode replay for brax environments.
    """
    env_name = str(
        np.loadtxt(os.path.join(load_dir, "episode", "env_name.txt"), dtype="str")
    )
    qp_pos = np.load(os.path.join(load_dir, "episode", "qp_pos.npy"))
    qp_rot = np.load(os.path.join(load_dir, "episode", "qp_rot.npy"))
    qp_vel = np.load(os.path.join(load_dir, "episode", "qp_vel.npy"))
    qp_ang = np.load(os.path.join(load_dir, "episode", "qp_ang.npy"))
    env = envs.create(env_name=env_name)
    episode_length = len(qp_pos)
    episode = [
        QP(qp_pos[i], qp_rot[i], qp_vel[i], qp_ang[i]) for i in range(episode_length)
    ]
    display.display(HTML(html.render(env.sys, episode)))
