import haiku as hk
import jax.numpy as jnp
from jax import nn

from xpag.agents.rljax_agents.network.base import MLP


class DeterministicPolicy(hk.Module):
    """
    Policy for DDPG and TD3.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(256, 256),
        d2rl=False,
    ):
        super(DeterministicPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.d2rl = d2rl

    def __call__(self, x):
        return MLP(
            self.action_dim,
            self.hidden_units,
            hidden_activation=nn.relu,
            output_activation=jnp.tanh,
            d2rl=self.d2rl,
        )(x)


class StateDependentGaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        clip_log_std=True,
        d2rl=False,
    ):
        super(StateDependentGaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clip_log_std = clip_log_std
        self.d2rl = d2rl

    def __call__(self, x):
        x = MLP(
            2 * self.action_dim,
            self.hidden_units,
            hidden_activation=nn.relu,
            d2rl=self.d2rl,
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        if self.clip_log_std:
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                jnp.tanh(log_std) + 1.0
            )
        return mean, log_std


class StateIndependentGaussianPolicy(hk.Module):
    """
    Policy for PPO.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(64, 64),
    ):
        super(StateIndependentGaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units

    def __call__(self, x):
        mean = MLP(
            self.action_dim,
            self.hidden_units,
            hidden_activation=jnp.tanh,
            output_scale=0.01,
        )(x)
        log_std = hk.get_parameter("log_std", (1, self.action_dim), init=jnp.zeros)
        return mean, log_std
