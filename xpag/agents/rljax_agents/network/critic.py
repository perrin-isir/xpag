import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn

from xpag.agents.rljax_agents.network.base import MLP


class ContinuousVFunction(hk.Module):
    """
    Critic for PPO.
    """

    def __init__(
        self,
        num_critics=1,
        hidden_units=(64, 64),
    ):
        super(ContinuousVFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, x):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=jnp.tanh,
            )(x)

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQFunction(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units=(256, 256),
        d2rl=False,
    ):
        super(ContinuousQFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQuantileFunction(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
        self,
        num_critics=5,
        hidden_units=(512, 512, 512),
        num_quantiles=25,
        d2rl=False,
    ):
        super(ContinuousQuantileFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.num_quantiles = num_quantiles
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                self.num_quantiles,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        return [_fn(x) for _ in range(self.num_critics)]
