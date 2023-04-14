# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
from typing import List, Dict, Any
from xpag.tools.utils import DataType, datatype_convert
from matplotlib import figure
from matplotlib import collections as mc


def _from_1d_to_2d(t, v):
    assert len(v) == 1 or len(v) == 2, "projection function outputs must be 1D or 2D"
    if len(v) == 2:
        return 2, v
    else:
        return 1, np.array([t, v[0]])


def _expand_bounds(bounds):
    bmin = bounds[0]
    bmax = bounds[1]
    expand_ratio = 1e-1
    min_expansion = 1e-3
    delta = max((bmax - bmin) * expand_ratio, min_expansion)
    return [bmin - delta, bmax + delta]


def single_episode_plot(
    filename: str,
    step_list: List[Dict[str, Any]],
    projection_function=lambda x: x[0:2],
    plot_env_function=None,
):
    """Plots an episode, using a 1D or 2D projection from observations, or
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
    projection_dimension = None
    for t, step in enumerate(step_list):
        if (
            isinstance(step["observation"], dict)
            and "achieved_goal" in step["observation"]
        ):
            goalenv = True
            projection_dimension, x_obs = _from_1d_to_2d(
                t,
                projection_function(
                    datatype_convert(
                        step["observation"]["achieved_goal"][0], DataType.NUMPY
                    )
                ),
            )
            projection_dimension, x_obs_next = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(
                        step["next_observation"]["achieved_goal"][0], DataType.NUMPY
                    )
                ),
            )
            projection_dimension, gxy = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(
                        step["observation"]["desired_goal"][0], DataType.NUMPY
                    )
                ),
            )
            gx.append(gxy[0])
            xmax = max(xmax, gxy[0])
            xmin = min(xmin, gxy[0])
            gy.append(gxy[1])
            ymax = max(ymax, gxy[1])
            ymin = min(ymin, gxy[1])
        else:
            projection_dimension, x_obs = _from_1d_to_2d(
                t,
                projection_function(
                    datatype_convert(step["observation"][0], DataType.NUMPY)
                ),
            )
            projection_dimension, x_obs_next = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(step["next_observation"][0], DataType.NUMPY)
                ),
            )
        lines.append((x_obs, x_obs_next))
        xmax = max(xmax, max(x_obs[0], x_obs_next[0]))
        xmin = min(xmin, min(x_obs[0], x_obs_next[0]))
        ymax = max(ymax, max(x_obs[1], x_obs_next[1]))
        ymin = min(ymin, min(x_obs[1], x_obs_next[1]))
        rgbs.append(
            (1.0 - t / episode_length / 2.0, 0.2, 0.2 + t / episode_length / 2.0, 1)
        )
    ax.set_xlim(_expand_bounds([xmin, xmax]))
    ax.set_ylim(_expand_bounds([ymin, ymax]))
    if plot_env_function is not None and projection_dimension == 2:
        plot_env_function(ax)
    if goalenv:
        if projection_dimension == 2:
            ax.scatter(gx, gy, s=10, c="green", alpha=0.8)
        else:
            g_gather = np.vstack((gx, gy)).transpose()
            g_lines = list(zip(g_gather[:-1], g_gather[1:]))
            ax.add_collection(
                mc.LineCollection(g_lines, colors="green", linewidths=2.0)
            )
    ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1.0))
    fig.savefig(filename, dpi=200)
    fig.clf()
    ax.cla()
