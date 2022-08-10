"""Models."""

import optax
from functools import partial
from jax import numpy as jnp
import haiku as hk

from .modules import LearnedFeatures, HybridEmbedding


class MatrixFactorization(hk.Module):
    """Generic matrix factorization."""

    def __init__(
            self, module, runtime, shape=(10, 10), logsumexp=False,
            optimizer=optax.adam(0.001), name=None):
        super().__init__(name=name)

        self.module = module(samples=shape[0], name="module")
        self.runtime = runtime(samples=shape[1], name="runtime")

        self.logsumexp = logsumexp
        self.shape = shape

    def __call__(self, x):
        """<module(i), runtime(j)>."""
        if x is None:
            return self._predict_full()
        else:
            return self._predict(x)

    def _predict(self, x):
        ui = self.module(x[:, 0])
        vj = self.runtime(x[:, 1])
        if self.logsumexp:
            return jnp.log(jnp.sum(jnp.exp(ui + vj), axis=1))
        else:
            return jnp.sum(ui * vj, axis=1)

    def _predict_full(self):
        module_emb = self.module(jnp.arange(self.shape[0]))
        runtime_emb = self.runtime(jnp.arange(self.shape[1]))
        if self.logsumexp:
            return jnp.log(jnp.sum(
                jnp.exp(
                    module_emb.reshape(self.shape[0], 1, -1)
                    + runtime_emb.reshape(1, self.shape[1], -1)),
                axis=2))
        else:
            return jnp.matmul(module_emb, runtime_emb.T)


def linear(dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: y_ij = <u_i, v_j>."""
    return MatrixFactorization(
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        shape=shape, logsumexp=False, name="linear")


def logsumexp(dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: y_ij = log(sum(exp(u_i + v_j)))."""
    return MatrixFactorization(
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        shape=shape, logsumexp=True, name="linear")


def embedding(
        runtime_data=None, module_data=None,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Side Information using NN Embedding."""
    def _features(data):
        if data is None:
            return partial(LearnedFeatures, dim=layers[-1], scale=scale)
        else:
            return partial(
                HybridEmbedding, data, layers=layers, dim=dim, scale=scale)

    return MatrixFactorization(
        _features(runtime_data), _features(module_data),
        shape=shape, logsumexp=False, name="Embedding")
