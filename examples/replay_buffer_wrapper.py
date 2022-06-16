from examples.replay_buffer_brax import UniformSamplingQueue
from xpag.buffers import Buffer
from xpag.samplers import Sampler
from xpag.tools.utils import DeviceArray
from typing import Dict, Any, Union
import torch
import jax.numpy as jnp
import jax


class BraxUniformSamplingQueueToXpag(Buffer):
    def __init__(
        self,
        buffer_size: int,
        sampler: Sampler,
    ):
        self.buffer_size = buffer_size
        self.sampler = sampler
        self.replay_buffer = None
        self.buffer_state = None

    def init_rp_buffer(self, dummy_step, rng=0):
        """
        Init the replay buffer with a dummy step. (!! do not include the batch dimension !!)
        """
        self.replay_buffer = UniformSamplingQueue(self.buffer_size, dummy_step)
        self.buffer_state = self.replay_buffer.init(jax.random.PRNGKey(rng))
        self.replay_buffer.insert_jit = jax.jit(self.replay_buffer.insert)
        self.replay_buffer.sample_jit = jax.jit(
            self.replay_buffer.sample, static_argnames="batch_size"
        )

    def insert(self, step_batch: Dict[str, Any]):
        """Inserts a transition in the buffer"""
        assert self.buffer_state is not None, "you must call init_rp_buffer before the first insert"
        self.replay_buffer.insert_jit(self.buffer_state, step_batch)

    def pre_sample(self) -> Dict[str, Union[torch.Tensor, jnp.ndarray, DeviceArray]]:
        """Returns a part of the buffer from which the sampler will extract samples"""
        raise Exception("Not implemented")

    def sample(
        self, batch_size
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray, DeviceArray]]:
        """Returns a batch of transitions"""
        self.buffer_state, batch = self.replay_buffer.sample_jit(
            self.buffer_state, batch_size
        )
        return batch
