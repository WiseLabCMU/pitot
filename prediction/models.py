"""Models."""

import optax
from jax import numpy as jnp
import haiku as hk

from .modules import LearnedFeatures, HybridEmbedding


class MFBase(hk.Module):
    """Base class for matrix factorization methods."""

    optimizer = optax.adam(0.001)
    epochs = 25
    epoch_size = 100


class MFLinear(MFBase):
    """Linear matrix factorization: y_ij = <u_i, v_j>."""

    # optimizer = optax.adam(
    #     optax.piecewise_constant_schedule(0.1, {1000: 0.1, 5000: 0.1}))
    epoch_size = 20

    def __init__(self, dim=8, scale=0.01, samples=(10, 10), name=None):
        super().__init__(name=name)

        self.U = LearnedFeatures(dim, samples[0], scale=scale, name="modules")
        self.V = LearnedFeatures(dim, samples[1], scale=scale, name="runtimes")

    def __call__(self, x):
        """<u_i * v_j>."""
        ui = self.U(x[:, 0])
        vj = self.V(x[:, 1])
        return jnp.sum(ui * vj, axis=1)


class MFLogSumExp(MFLinear):
    """Linear matrix factorization: y_ij = logsumexp(u_i + v_j)."""

    def __call__(self, x):
        """logsumexp(u_i + v_j)."""
        ui = self.U(x[:, 0])
        vj = self.V(x[:, 1])
        return jnp.log(jnp.sum(jnp.exp(ui + vj), axis=1))


class MFEmbedding(MFBase):
    """Matrix Factorization with Side Information using NN Embedding."""

    def __init__(
            self, runtime_data, module_data, samples=(10, 10),
            layers=[64, 32], dim=4, scale=0.1, name=None):
        super().__init__(name=name)

        self.module = HybridEmbedding(
            module_data, layers=layers, dim=dim, samples=samples[0],
            scale=scale, name="module")
        self.runtime = HybridEmbedding(
            runtime_data, layers=layers, dim=dim, samples=samples[1],
            scale=scale, name="runtime")

    def __call__(self, x):
        """<module_emb(u_i, x_i), runtime_emb(v_j, y_j)>."""
        return jnp.sum(self.module(x[:, 0]) * self.runtime(x[:, 1]), axis=1)
