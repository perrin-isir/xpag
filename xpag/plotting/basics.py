import matplotlib.pyplot as plt
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
    # if plot_ax is None:
    #     _, ax = plt.subplots()
    # else:
    #     ax = plot_ax
    _, ax = plt.subplots()
    if plot_env_function is not None:
        plot_env_function(ax)
    for ep_idx in range(len(episode.obs)):
        lines = []
        rgbs = []
        for j in range(episode_length):
            x_obs = projection_function(
                datatype_convert(episode.obs[ep_idx][j], DataType.NUMPY))
            x_obs_next = projection_function(
                datatype_convert(episode.obs_next[ep_idx][j][0:2], DataType.NUMPY))
            lines.append((x_obs, x_obs_next))
            rgbs.append((1.0 - j / episode_length / 2.,
                         0.2,
                         0.2 + j / episode_length / 2.,
                         1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1.5))
    plt.savefig(filename, dpi=200)
    plt.close()


    # lenpaths = len(paths)
    #     for i, path in enumerate(paths):
    #         lines = []
    #         rgbs = []
    #         for p in path:
    #             lines.append((p["obs"], p["obs_next"]))
    #             rgbs.append((1.0 - i / lenpaths, 0.2, 0.2, 1))
    #         ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
    #     ax.set_xlim([-1, 1])
    #     ax.set_ylim([-1, 1])
    #     # plt.xlim(-1, 1)
    #     # plt.ylim(-1, 1)
    #     # plt.savefig(filename, dpi=200)
    #     # plt.close()
