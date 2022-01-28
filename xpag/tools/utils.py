from enum import Enum
from typing import Tuple, Union
import collections
import functools
import torch
import numpy as np
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


class DataType(Enum):
    TORCH = 'data represented as torch tensors'
    NUMPY = 'data represented as numpy arrays'
    JAX = 'data represented as jax DeviceArrays'


def define_step_data(is_goalenv: bool,
                     num_envs: int,
                     observation_dim: int,
                     achieved_goal_dim: Union[int, None],
                     desired_goal_dim: Union[int, None],
                     action_dim: int,
                     episode_max_length: int,
                     datatype: DataType = DataType.TORCH,
                     device: str = 'cpu') -> Tuple[collections.namedtuple,
                                                   collections.namedtuple]:
    if is_goalenv:
        fields = ('obs', 'obs_next', 'ag', 'ag_next', 'g', 'g_next', 'actions',
                  'terminals')
        sizes = [observation_dim, observation_dim, achieved_goal_dim, achieved_goal_dim,
                 desired_goal_dim, desired_goal_dim, action_dim, 1]
    else:
        fields = ('obs', 'obs_next', 'actions', 'r', 'terminals')
        sizes = [observation_dim, observation_dim, action_dim, 1, 1]

    empty_func = None
    if datatype == DataType.TORCH:
        empty_func = functools.partial(torch.empty, device=device)
    elif datatype == DataType.NUMPY:
        empty_func = np.empty

    def defaults(k):
        return [empty_func((k, episode_max_length, siz)) for siz in sizes]

    return collections.namedtuple(
        'StepDataUnique', fields, defaults=defaults(1)
    ), collections.namedtuple(
        'StepDataMultiple', fields, defaults=defaults(num_envs)
    )


def register_step_in_episode(
        episode,
        episode_t,
        is_goalenv,
        num_envs,
        o,
        action,
        new_o,
        reward,
        done):
    episode.terminals[:, episode_t, :] = done
    episode.actions[:, episode_t, :] = reshape_func(
        action, (num_envs, episode.actions.shape[-1]))
    if is_goalenv:
        episode.obs[:, episode_t, :] = reshape_func(
            o['observation'], (num_envs, episode.obs.shape[-1]))
        episode.obs_next[:, episode_t, :] = reshape_func(
            new_o['observation'], (num_envs, episode.obs_next.shape[-1]))
        episode.ag[:, episode_t, :] = reshape_func(
            o['achieved_goal'], (num_envs, episode.ag.shape[-1]))
        episode.ag_next[:, episode_t, :] = reshape_func(
            new_o['achieved_goal'], (num_envs, episode.ag_next.shape[-1]))
        episode.g[:, episode_t, :] = reshape_func(
            o['desired_goal'], (num_envs, episode.g.shape[-1]))
        episode.g_next[:, episode_t, :] = reshape_func(
            new_o['desired_goal'], (num_envs, episode.g_next.shape[-1]))
    else:
        episode.r[:, episode_t, :] = reward
        episode.obs[:, episode_t, :] = reshape_func(
            o, (num_envs, episode.obs.shape[-1]))
        episode.obs_next[:, episode_t, :] = reshape_func(
            new_o, (num_envs, episode.obs_next.shape[-1]))


def step_data_select(sd_one: collections.namedtuple, sd_m: collections.namedtuple,
                     i: int):
    for f in sd_one._fields:
        sd_one._asdict()[f][0] = sd_m._asdict()[f][i]


def reshape_func(
        x: Union[torch.Tensor, np.ndarray, DeviceArray, list, float, np.float],
        shape: Tuple[int, ...]
) -> Union[torch.Tensor, np.ndarray]:
    if torch.is_tensor(x) or type(x) == np.ndarray or type(x) == DeviceArray:
        return x.reshape(shape)
    else:
        return np.array(x).reshape(shape)


def hstack_func(
        x: Union[torch.Tensor, np.ndarray, DeviceArray],
        y: Union[torch.Tensor, np.ndarray, DeviceArray]
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return torch.hstack((x, y))
    elif type(x) == DeviceArray and type(y) == DeviceArray:
        return jnp.hstack((x, y))
    else:
        return np.hstack((x, y))


def max_func(
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return torch.maximum(x, y)
    elif type(x) == DeviceArray and type(y) == DeviceArray:
        return jnp.maximum((x, y))
    else:
        return np.maximum(x, y)


def datatype_convert(
        x: Union[torch.Tensor, np.ndarray, DeviceArray, list, float, np.float],
        datatype: DataType = DataType.TORCH,
        device: str = 'cpu'
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if datatype == DataType.TORCH:
        if torch.is_tensor(x):
            return x.to(device=device)
        elif type(x) == DeviceArray:
            return torch.tensor(np.array(x), device=device)
        else:
            return torch.tensor(x, device=device)
    elif datatype == DataType.NUMPY:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)
    elif datatype == DataType.JAX:
        if torch.is_tensor(x):
            return jnp.array(x.detach().cpu().numpy())
        elif type(x) == DeviceArray:
            return x
        else:
            return jnp.array(x)
