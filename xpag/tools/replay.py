import os
import numpy as np
import gym
from IPython import display
from PIL import Image, ImageDraw
import ipywidgets
from typing import Callable
import base64
import hashlib
import io


class DownloadButton(ipywidgets.Button):
    """Download button with dynamic content

    The content is generated using a callback when the button is clicked.
    """

    # def __init__(self, filename: str, contents: Callable[[], str], **kwargs):
    def __init__(self, filename: str, contents: Callable[[], bytes], **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.filename = filename
        self.contents = contents
        self.on_click(self.__on_click)

    def __on_click(self, b):
        # contents: bytes = self.contents().encode('utf-8')
        contents: bytes = self.contents()
        b64 = base64.b64encode(contents)
        payload = b64.decode()
        digest = hashlib.md5(contents).hexdigest()  # bypass browser cache
        id_ = f"dl_{digest}"

        display.display(
            display.HTML(
                f"""
<html>
<body>
<a id="{id_}" download="{self.filename}" href="data:text/csv;base64,{payload}" download>
</a>

<script>
(function download() {{
document.getElementById('{id_}').click();
}})()
</script>

</body>
</html>
"""
            )
        )


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

    img_dict = {}

    play = ipywidgets.Play(
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
    display_sequence(slider)

    def create_gif():
        f = io.BytesIO()
        img_dict[0].save(
            f,
            format="png",
            append_images=[img_dict[0]],
            save_all=True,
            duration=env_replay.model.opt.timestep * 1000,
            loop=0,
        )
        return f.getvalue()

    display.display(
        DownloadButton(filename="foo.gif", contents=create_gif, description="download")
    )
