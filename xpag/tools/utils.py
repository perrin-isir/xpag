# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from enum import Enum
from typing import Tuple, Union, Dict
import torch
import numpy as np
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


class DataType(Enum):
    TORCH_CPU = "data represented as torch tensors on CPU"
    TORCH_CUDA = "data represented as torch tensors on GPU"
    NUMPY = "data represented as numpy arrays"
    JAX = "data represented as jax DeviceArrays"


def reshape(
    x: Union[torch.Tensor, np.ndarray, DeviceArray, list, float],
    shape: Tuple[int, ...],
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if torch.is_tensor(x) or type(x) == np.ndarray or type(x) == DeviceArray:
        return x.reshape(shape)
    else:
        return np.array(x).reshape(shape)


def hstack(
    x: Union[torch.Tensor, np.ndarray, DeviceArray],
    y: Union[torch.Tensor, np.ndarray, DeviceArray],
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return torch.hstack((x, y))
    elif type(x) == DeviceArray and type(y) == DeviceArray:
        return jnp.hstack((x, y))
    else:
        return np.hstack((x, y))


def maximum(
    x: Union[torch.Tensor, np.ndarray, DeviceArray],
    y: Union[torch.Tensor, np.ndarray, DeviceArray],
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return torch.maximum(x, y)
    elif type(x) == DeviceArray and type(y) == DeviceArray:
        return jnp.maximum(x, y)
    else:
        return np.maximum(x, y)


def squeeze(
    x: Union[torch.Tensor, np.ndarray, DeviceArray]
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if torch.is_tensor(x):
        return torch.squeeze(x)
    elif type(x) == DeviceArray:
        return jnp.squeeze(x)
    else:
        return np.squeeze(x)


def datatype_convert(
    x: Union[torch.Tensor, np.ndarray, DeviceArray, list, float],
    datatype: DataType = DataType.NUMPY,
) -> Union[torch.Tensor, np.ndarray, DeviceArray]:
    if datatype == DataType.TORCH_CPU or datatype == DataType.TORCH_CUDA:
        if torch.is_tensor(x):
            if datatype == DataType.TORCH_CPU:
                return x.to(device="cpu")
            else:
                return x.to(device="cuda")
        elif type(x) == DeviceArray:
            if datatype == DataType.TORCH_CPU:
                return torch.tensor(np.array(x), device="cpu")
            else:
                return torch.tensor(np.array(x), device="cuda")
        else:
            if datatype == DataType.TORCH_CPU:
                return torch.tensor(x, device="cpu")
            else:
                return torch.tensor(x, device="cuda")
    elif datatype == DataType.NUMPY:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif type(x) == np.ndarray:
            return x
        else:
            return np.array(x)
    elif datatype == DataType.JAX:
        if torch.is_tensor(x):
            return jnp.array(x.detach().cpu().numpy())
        elif type(x) == DeviceArray:
            return x
        else:
            return jnp.array(x)


def get_env_dimensions(info: dict, is_goalenv: bool, env) -> Dict[str, int]:
    """
    Return the important dimensions associated with an environment (observation_dim,
    action_dim, ...)
    """
    is_goalenv = is_goalenv
    if hasattr(env, "is_vector_env"):
        gymvecenv = env.is_vector_env
    else:
        gymvecenv = False
    dims = {}
    if gymvecenv:
        info["action_dim"] = env.single_action_space.shape[-1]
        info["observation_dim"] = (
            env.single_observation_space["observation"].shape[-1]
            if is_goalenv
            else env.single_observation_space.shape[-1]
        )
        info["achieved_goal_dim"] = (
            env.single_observation_space["achieved_goal"].shape[-1]
            if is_goalenv
            else None
        )
        info["desired_goal_dim"] = (
            env.single_observation_space["desired_goal"].shape[-1]
            if is_goalenv
            else None
        )
    else:
        info["action_dim"] = env.action_space.shape[-1]
        info["observation_dim"] = (
            env.observation_space["observation"].shape[-1]
            if is_goalenv
            else env.observation_space.shape[-1]
        )
        info["achieved_goal_dim"] = (
            env.observation_space["achieved_goal"].shape[-1] if is_goalenv else None
        )
        info["desired_goal_dim"] = (
            env.observation_space["desired_goal"].shape[-1] if is_goalenv else None
        )
    return dims
