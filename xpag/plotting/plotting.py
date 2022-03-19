# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from matplotlib import figure
import numpy as np
from typing import List, Dict, Any
from matplotlib import collections as mc
from xpag.tools.utils import DataType, datatype_convert


def single_episode_plot(
    filename: str,
    step_list: List[Dict[str, Any]],
    projection_function=lambda x: x[0:2],
    plot_env_function=None,
):
    """Plots an episode, using a 2D projection from observations, or
    from achieved and desired goals in the case of GoalEnv environments.
    """
    fig = figure.Figure()
    ax = fig.subplots(1)
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    lines = []
    rgbs = []
    gx = []
    gy = []
    episode_length = len(step_list)
    goalenv = False
    for j, step in enumerate(step_list):
        if (
            isinstance(step["observation"], dict)
            and "achieved_goal" in step["observation"]
        ):
            goalenv = True
            x_obs = projection_function(
                datatype_convert(
                    step["observation"]["achieved_goal"][0], DataType.NUMPY
                )
            )
            x_obs_next = projection_function(
                datatype_convert(
                    step["next_observation"]["achieved_goal"][0], DataType.NUMPY
                )
            )
            gxy = projection_function(
                datatype_convert(step["observation"]["desired_goal"][0], DataType.NUMPY)
            )
            gx.append(gxy[0])
            gy.append(gxy[1])
        else:
            x_obs = projection_function(
                datatype_convert(step["observation"][0], DataType.NUMPY)
            )
            x_obs_next = projection_function(
                datatype_convert(step["next_observation"][0], DataType.NUMPY)
            )
        lines.append((x_obs, x_obs_next))
        xmax = max(xmax, max(x_obs[0], x_obs_next[0]))
        xmin = min(xmin, min(x_obs[0], x_obs_next[0]))
        ymax = max(ymax, max(x_obs[1], x_obs_next[1]))
        ymin = min(ymin, min(x_obs[1], x_obs_next[1]))
        rgbs.append(
            (1.0 - j / episode_length / 2.0, 0.2, 0.2 + j / episode_length / 2.0, 1)
        )
    ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1.0))
    if goalenv:
        ax.scatter(gx, gy, s=10, c="green", alpha=0.8)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if plot_env_function is not None:
        plot_env_function(ax)
    fig.savefig(filename, dpi=200)
    fig.clf()
    ax.cla()
