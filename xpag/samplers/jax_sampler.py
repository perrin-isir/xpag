# Part of this file was taken from :
# https://github.com/google/brax/blob/main/brax/training/replay_buffers.py

# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Generic, Tuple, Union, TypeVar
import jax
import flax
from jax import flatten_util
import jax.numpy as jnp
from xpag.samplers.sampler import Sampler


PRNGKey = jnp.ndarray
Sample = TypeVar("Sample")


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
    current_position: jnp.ndarray
    current_size: jnp.ndarray
    key: PRNGKey


class RBUniformSampling(Generic[Sample]):
    """Uniform sampling in the Replay buffer.
    * It performs uniform random sampling with replacement of a batch of size
      `batch_size`
    """

    def __init__(self, dummy_data_sample: Sample):
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)

    def sample(
        self, buffer_state: ReplayBufferState, batch_size: int
    ) -> Tuple[ReplayBufferState, jnp.ndarray]:
        """Sample a batch of data.
        Args:
          buffer_state: Buffer state
          batch_size: size of the batch
        Returns:
          New buffer state and a batch with leading dimension 'sample_batch_size'.
        """
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key, (batch_size,), minval=0, maxval=buffer_state.current_size
        )
        batch = jnp.take(buffer_state.data, idx, axis=0, mode="clip")
        return buffer_state.replace(key=key), self._unflatten_fn(batch)


class JaxSampler(Sampler):
    def __init__(self, *, seed: Union[int, None] = None):
        self.sampler = None
        self.rbus = None
        self.sample_jit = None
        super().__init__(seed=seed)

    def init(self, dummy_step):
        self.rbus = RBUniformSampling(dummy_step)
        self.sample_jit = jax.jit(self.rbus.sample, static_argnames="batch_size")

    def sample(
        self,
        buffer_state,
        batch_size: int,
    ) -> jnp.ndarray:
        if self.rbus is None:
            raise RuntimeError("init() must be called before sample().")
        else:
            return self.rbus.sample(buffer_state, batch_size)
