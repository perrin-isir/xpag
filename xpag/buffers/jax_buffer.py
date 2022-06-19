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

from typing import Generic
import jax
from jax import flatten_util
import jax.numpy as jnp
from xpag.buffers.buffer import Buffer
from xpag.samplers.jax_sampler import JaxSampler, PRNGKey, Sample, ReplayBufferState
from typing import Dict, Any


class RBQueue(Generic[Sample]):
    """Replay buffer.
    * It behaves as a limited size queue (if buffer is full it removes the oldest
      elements when new one is inserted).
    * It supports batch insertion only (no single element)
    """

    def __init__(self, max_replay_size: int, dummy_data_sample: Sample):
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        data_size = len(dummy_flatten)
        self._data_shape = (max_replay_size, data_size)
        self._data_dtype = dummy_flatten.dtype

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            current_size=jnp.zeros((), jnp.int32),
            current_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(
        self, buffer_state: ReplayBufferState, samples: Sample
    ) -> ReplayBufferState:
        """Insert data in the replay buffer.
        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.
        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # Make sure update is not larger than the maximum replay size.
        if len(update) > len(data):
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {len(update)}, max replay size {len(data)}"
            )

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.current_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % len(data)
        size = jnp.minimum(buffer_state.current_size + len(update), len(data))

        return ReplayBufferState(
            data=data,
            current_position=position,
            current_size=size,
            key=buffer_state.key,
        )

    @staticmethod
    def size(buffer_state: ReplayBufferState) -> int:
        return int(buffer_state.current_size)


class JaxBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int,
        sampler: JaxSampler,
    ):
        super().__init__(buffer_size, sampler)
        self.replay_buffer = None
        self.buffer_state = None
        self.initialized = False

    def init_rp_buffer(self, dummy_step, rng: int):
        """
        Init the replay buffer with a dummy step.
        (!! do not include the batch dimension !!)
        """
        self.replay_buffer = RBQueue(self.buffer_size, dummy_step)
        self.buffer_state = self.replay_buffer.init(jax.random.PRNGKey(rng))
        self.replay_buffer.insert_jit = jax.jit(self.replay_buffer.insert)

    def insert(self, step_batch: Dict[str, Any]):
        """Inserts a transition in the buffer"""
        if not self.initialized:
            dummy_step = {}
            for key in step_batch.keys():
                dummy_step[key] = jnp.zeros(step_batch[key].shape[1])
            self.init_rp_buffer(
                dummy_step, 0 if self.sampler.seed is None else self.sampler.seed
            )
            self.sampler.init(dummy_step)
            self.initialized = True
        self.buffer_state = self.replay_buffer.insert_jit(self.buffer_state, step_batch)

    def sample(self, batch_size) -> Dict[str, jnp.ndarray]:
        """Returns a batch of transitions"""
        self.buffer_state, batch = self.sampler.sample_jit(
            self.buffer_state, batch_size
        )
        return batch
