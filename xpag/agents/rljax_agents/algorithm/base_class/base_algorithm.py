from abc import ABC, abstractmethod
from functools import partial

import jax
import numpy as np
from haiku import PRNGSequence

from xpag.agents.rljax_agents.util import soft_update


class BaseAlgorithm(ABC):
    """
    Base class for algorithms.
    """

    name = None

    def __init__(
        self,
        num_agent_steps,
        observation_dim,
        action_dim,
        seed,
        max_grad_norm,
        gamma,
    ):
        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.discrete_action = False

    def get_mask(self, env, done):
        return done if (self.episode_step != env._max_episode_steps) else False

    def get_key_list(self, num_keys):
        return [next(self.rng) for _ in range(num_keys)]

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def update(self, writer):
        pass

    @abstractmethod
    def save_params(self, save_dir):
        pass

    @abstractmethod
    def load_params(self, save_dir):
        pass

    def __str__(self):
        return self.name


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        observation_dim,
        action_dim,
        seed,
        max_grad_norm,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        assert update_interval_target or tau
        super(OffPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            observation_dim=observation_dim,
            action_dim=action_dim,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        if not hasattr(self, "buffer"):
            self.buffer = None

        self.discount = gamma**nstep
        self.use_per = use_per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target

        if update_interval_target:
            self._update_target = jax.jit(partial(soft_update, tau=1.0))
        else:
            self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self):
        return (
            self.agent_step % self.update_interval == 0
            and self.agent_step >= self.start_steps
        )

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        mask = self.get_mask(env, done)
        self.buffer.append(state, action, reward, mask, next_state, done)

        if done:
            self.episode_step = 0
            next_state, _ = env.reset()

        return next_state
