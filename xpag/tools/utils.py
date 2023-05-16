# Copyright 2022-2023, CNRS.
#
# Licensed under the BSD 3-Clause License.

from enum import Enum
from typing import Tuple, Union, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces


class DataType(Enum):
    NUMPY = "data represented as numpy arrays"
    JAX = "data represented as jax.numpy arrays"


def get_datatype(x: Union[np.ndarray, jnp.ndarray]) -> DataType:
    if isinstance(x, jnp.ndarray):
        return DataType.JAX
    elif isinstance(x, np.ndarray):
        return DataType.NUMPY
    else:
        raise TypeError(f"{type(x)} not handled.")


def datatype_convert(
    x: Union[np.ndarray, jnp.ndarray, list, float],
    datatype: Union[DataType, None] = DataType.NUMPY,
) -> Union[np.ndarray, jnp.ndarray]:
    if datatype is None:
        return x
    elif datatype == DataType.NUMPY:
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
    elif datatype == DataType.JAX:
        if isinstance(x, jnp.ndarray):
            return x
        else:
            return jnp.array(x)


def reshape(
    x: Union[np.ndarray, jnp.ndarray, list, float],
    shape: Tuple[int, ...],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
        return x.reshape(shape)
    else:
        return np.array(x).reshape(shape)


def hstack(
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.hstack((x, y))
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.hstack((x, y))
    else:
        raise TypeError("Incorrect or non-matching input types.")


def logical_or(
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.logical_or(x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.logical_or(x, y)
    else:
        raise TypeError("Incorrect or non-matching input types.")


def maximum(
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.maximum(x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.maximum(x, y)
    else:
        raise TypeError("Incorrect or non-matching input types.")


def squeeze(x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray):
        return jnp.squeeze(x)
    else:
        return np.squeeze(x)


def where(
    condition: Any,
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        return jnp.where(condition, x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.where(condition, x, y)
    else:
        raise TypeError("Incorrect or non-matching input types.")


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
        if is_goalenv:
            assert (
                isinstance(env.single_observation_space["observation"], spaces.box.Box)
                and isinstance(
                    env.single_observation_space["achieved_goal"], spaces.box.Box
                )
                and isinstance(
                    env.single_observation_space["desired_goal"], spaces.box.Box
                )
                and isinstance(env.single_action_space, spaces.box.Box)
            ), (
                'env.single_observation_space["observation"] and '
                'env.single_observation_space["achieved_goal"] and '
                'env.single_observation_space["desired_goal"] and '
                "env.single_action_space must be of type gymnasium.spaces.box.Box"
            )
        else:
            assert isinstance(
                env.single_observation_space, spaces.box.Box
            ) and isinstance(env.single_action_space, spaces.box.Box), (
                "env.single_observation_space and "
                "env.single_action_space must be of type gymnasium.spaces.box.Box"
            )

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
        if is_goalenv:
            assert (
                isinstance(env.observation_space["observation"], spaces.box.Box)
                and isinstance(env.observation_space["achieved_goal"], spaces.box.Box)
                and isinstance(env.observation_space["desired_goal"], spaces.box.Box)
                and isinstance(env.action_space, spaces.box.Box)
            ), (
                'env.observation_space["observation"] and '
                'env.observation_space["achieved_goal"] and '
                'env.observation_space["desired_goal"] and '
                "env.action_space must be of type gymnasium.spaces.box.Box"
            )
        else:
            assert isinstance(env.observation_space, spaces.box.Box) and isinstance(
                env.action_space, spaces.box.Box
            ), (
                "env.observation_space and "
                "env.action_space must be of type gymnasium.spaces.box.Box"
            )

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


def tree_sum(tree: Any):
    elt_list = jax.tree_util.tree_flatten(tree)[0]
    cumsum = 0.0
    for elt in elt_list:
        cumsum += elt.sum()
    return cumsum
