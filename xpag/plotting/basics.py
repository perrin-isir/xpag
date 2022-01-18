import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
import collections
from xpag.tools.utils import DataType, datatype_convert


def plot_episode_2d(filename: str,
                    episode: collections.namedtuple,
                    episode_length: int,
                    projection_function=lambda x: x[0:2],
                    plot_env_function=None,
                    ):
    """Plot episode(s), using a 2D projection from observations.
    It can plot multiple episodes, but they must have the same length.
    """
    assert (len(episode.obs.shape) == 3)
    _, ax = plt.subplots()
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    for ep_idx in range(len(episode.obs)):
        lines = []
        rgbs = []
        for j in range(episode_length):
            x_obs = projection_function(
                datatype_convert(episode.obs[ep_idx][j], DataType.NUMPY))
            x_obs_next = projection_function(
                datatype_convert(episode.obs_next[ep_idx][j], DataType.NUMPY))
            lines.append((x_obs, x_obs_next))
            xmax = max(xmax, max(x_obs[0], x_obs_next[0]))
            xmin = min(xmin, min(x_obs[0], x_obs_next[0]))
            ymax = max(ymax, max(x_obs[1], x_obs_next[1]))
            ymin = min(ymin, min(x_obs[1], x_obs_next[1]))
            rgbs.append((1.0 - j / episode_length / 2.,
                         0.2,
                         0.2 + j / episode_length / 2.,
                         1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1.5))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    if plot_env_function is not None:
        plot_env_function(ax)
    plt.savefig(filename, dpi=200)
    plt.close()
