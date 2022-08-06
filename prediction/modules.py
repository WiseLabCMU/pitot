"""Matrix factorization components."""

import jax
from jax import numpy as jnp
import haiku as hk


class SideInformation(hk.Module):
    """Lookup table for matrix factorization side information."""

    def __init__(self, data, name=None):
        super().__init__(name=name)
        self.data = data

    def __call__(self, i):
        """Index into side info with shape (index, features)."""
        return self.data[i]


class LearnedFeatures(hk.Module):
    """Lookup table for learned matrix factorization."""

    def __init__(self, dim=8, samples=10, scale=1.0, name=None):
        super().__init__(name=name)
        self.dim = (samples, dim)
        self.scale = scale

    def __call__(self, i):
        """Index into weights with shape (index, features)."""
        init = hk.initializers.RandomUniform(minval=0, maxval=self.scale)
        X = hk.get_parameter("X", shape=self.dim, init=init)
        return X[i]


def simple_mlp(layers, activation, name=None):
    """Create simple MLP."""
    _layers = []
    for i, out in enumerate(layers[:-1]):
        _layers.append(hk.Linear(out, name="layer_{}".format(i)))
        _layers.append(activation)
    _layers.append(hk.Linear(layers[-1], name="layer_out"))
    return hk.Sequential(_layers, name=name)


class HybridEmbedding(hk.Module):
    """Embedding with side information and some learned features."""

    def __init__(
            self, data, layers=[64, 32],
            dim=4, samples=10, scale=1.0, name=None):
        super().__init__(name=name)

        self.data = SideInformation(data, name="data")
        self.learned = LearnedFeatures(
            dim=dim, samples=samples, scale=scale, name="learned")
        self.embedding = simple_mlp(layers, jax.nn.tanh, name="embedding")

    def __call__(self, i):
        """MLP(data[i], features[i])."""
        return self.embedding(jnp.concatenate(
            [self.data(i), self.learned(i)], axis=1))
        # return self.embedding(self.data(i))
