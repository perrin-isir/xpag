from collections import deque
import numpy as np
from xpag.samplers.sampler import Sampler
from xpag.buffers.buffer import Buffer
from typing import Dict, Any, Optional


class NStepBuffer:
    """
    Buffer for calculating n-step returns.
    """

    def __init__(
        self,
        gamma=0.99,
        nstep=1,
    ):
        self.discount = [gamma**i for i in range(nstep)]
        self.nstep = nstep
        self.state = deque(maxlen=self.nstep)
        self.action = deque(maxlen=self.nstep)
        self.reward = deque(maxlen=self.nstep)

    def append(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)

    def get(self):
        assert len(self.reward) > 0

        state = self.state.popleft()
        action = self.action.popleft()
        reward = self.nstep_reward()
        return state, action, reward

    def nstep_reward(self):
        reward = np.sum([r * d for r, d in zip(self.reward, self.discount)])
        self.reward.popleft()
        return reward

    def is_empty(self):
        return len(self.reward) == 0

    def is_full(self):
        return len(self.reward) == self.nstep

    def __len__(self):
        return len(self.reward)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(
        self,
        buffer_size,
        observation_dim,
        action_dim,
        gamma,
        nstep,
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.nstep = nstep
        self.observation_dim = observation_dim

        self.state = np.empty((buffer_size, observation_dim), dtype=np.float32)
        self.next_state = np.empty((buffer_size, observation_dim), dtype=np.float32)

        self.action = np.empty((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.terminated = np.empty((buffer_size, 1), dtype=np.float32)
        self.truncated = np.empty((buffer_size, 1), dtype=np.float32)

        if nstep != 1:
            self.nstep_buffer = NStepBuffer(gamma, nstep)

    def append(self, state, action, reward, terminated, truncated, next_state):

        if self.nstep != 1:
            self.nstep_buffer.append(state, action, reward)

            if self.nstep_buffer.is_full():
                state, action, reward = self.nstep_buffer.get()
                self._append(state, action, reward, terminated, truncated, next_state)

            if terminated or truncated:
                while not self.nstep_buffer.is_empty():
                    state, action, reward = self.nstep_buffer.get()
                    self._append(
                        state, action, reward, terminated, truncated, next_state
                    )

        else:
            self._append(state, action, reward, terminated, truncated, next_state)

    def _append(self, state, action, reward, terminated, truncated, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.terminated[self._p] = float(terminated)
        self.truncated[self._p] = float(truncated)
        self.next_state[self._p] = next_state
        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def _sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample(self, idxes):
        state = self.state[idxes]
        next_state = self.next_state[idxes]

        return (
            state,
            self.action[idxes],
            self.reward[idxes],
            self.terminated[idxes],
            self.truncated[idxes],
            next_state,
        )

    def sample(self, batch_size):
        idxes = self._sample_idx(batch_size)
        batch = self._sample(idxes)
        # Use fake weight to use the same interface with PER.
        weight = np.ones((), dtype=np.float32)
        return weight, batch


class RljaxBuffer(Buffer):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        buffer_size: int,
        sampler: Optional[Sampler] = None,
        gamma: float = 0.99,
        nstep: int = 1,
    ):
        self.buffer = ReplayBuffer(
            buffer_size,
            observation_dim,
            action_dim,
            gamma,
            nstep,
        )
        self.buffer_size = buffer_size
        self.sampler = sampler

    def insert(self, step: Dict[str, Any]):
        """Inserts a transition in the buffer"""
        # __import__("IPython").embed()
        length = step["observation"].shape[0]
        for i in range(length):
            self.buffer.append(
                step["observation"][i],
                step["action"][i],
                step["reward"][i],
                step["terminated"][i],
                step["truncated"][i],
                step["next_observation"][i],
            )

    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        """Uses the sampler to returns a batch of transitions"""
        _, batch = self.buffer.sample(batch_size)
        return {
            "observation": batch[0],
            "action": batch[1],
            "reward": batch[2],
            "terminated": batch[3],
            "truncated": batch[4],
            "next_observation": batch[5],
        }
