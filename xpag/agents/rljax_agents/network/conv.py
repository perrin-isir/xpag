import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn


class DQNBody(hk.Module):
    """
    CNN for the atari environment.
    """

    def __init__(self):
        super(DQNBody, self).__init__()

    def __call__(self, x):
        # He's initializer.
        w_init = hk.initializers.Orthogonal(scale=np.sqrt(2))
        # Floatify the image.
        x = x.astype(jnp.float32) / 255.0
        # Apply CNN.
        x = hk.Conv2D(32, kernel_shape=8, stride=4, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=4, stride=2, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        # Flatten the feature map.
        return hk.Flatten()(x)
