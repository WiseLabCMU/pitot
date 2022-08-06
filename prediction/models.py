"""Models."""

import jax
import optax
from jax import numpy as jnp
import haiku as hk

from .modules import SideInformation, LearnedFeatures


class MFBase(hk.Module):
    """Base class for matrix factorization methods."""

    optimizer = optax.adam(0.001)
    epochs = 100
    epoch_size = 200


class MFLinear(MFBase):
    """Linear matrix factorization: y_ij = <u_i, v_j>."""

    optimizer = optax.adam(
        optax.piecewise_constant_schedule(0.1, {1000: 0.1, 5000: 0.1}))
    epochs = 100
    epoch_size = 100

    def __init__(self, dim=8, scale=1.0, samples=(10, 10), name=None):
        super().__init__(name=name)

        self.U = LearnedFeatures(8, samples[0], scale=scale, name="modules")
        self.V = LearnedFeatures(8, samples[1], scale=scale, name="runtimes")

    def __call__(self, x):
        """<u_i * v_j>."""
        ui = self.U(x[:, 0])
        vj = self.V(x[:, 1])
        return jnp.sum(ui * vj, axis=1)
