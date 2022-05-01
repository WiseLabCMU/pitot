"""Models."""

import jax
from jax import numpy as jnp
import haiku as hk


class FeatureLookup(hk.Module):
    """Lookup table for matrix completion."""

    def __init__(self, dim=(8, 8), samples=(10, 10), scale=0.01, name=None):
        super().__init__(name=name)
        self.Udim = (samples[0], dim[0])
        self.Vdim = (samples[1], dim[1])
        self.scale = scale

    def __call__(self, ij):
        init = hk.initializers.RandomUniform(minval=0, maxval=self.scale)
        
        U = hk.get_parameter("U", shape=self.Udim, init=init)
        V = hk.get_parameter("V", shape=self.Vdim, init=init)
        return U[ij[:, 0]], V[ij[:, 1]]


class MFLinear(hk.Module):
    """Linear matrix factorization: y_ij = <u_i, v_j>."""

    def __init__(self, dim=8, scale=1.0, samples=(10, 10), name=None):
        super().__init__(name=name)

        self.features = FeatureLookup(
            dim=(dim, dim),
            scale=jnp.sqrt(scale / dim), samples=samples, name='features')
    
    def __call__(self, x):

        ui, vj = self.features(x)
        return jnp.log(jnp.sum(jnp.exp(ui + vj), axis=1))


class MFNN(hk.Module):
    """Matrix factorization with neural network: y_ij = network(u_i, v_j)."""

    def __init__(
            self, dim=(8, 8), scale=1.0, samples=(10, 10),
            layers=[64, 32], name=None):
        super().__init__(name=name)

        scale = jnp.sqrt(scale * 2 / (dim[0] + dim[1]))
        self.features = FeatureLookup(
            dim=dim, scale=scale, samples=samples, name='features')
        self.layers = [
            hk.Linear(out, name='hidden_{}'.format(i))
            for i, out in enumerate(layers)]
        self.out = hk.Linear(1, name='output')

    def _forward(self, x):
        for layer in self.layers:
            x = jax.nn.sigmoid(layer(x))
        return self.out(x).reshape(-1)

    def __call__(self, x):
        ui, vj = self.features(x)
        x = jnp.concatenate([ui, vj], axis=1)
        return self._forward(x)


class MFNNSI(MFNN):
    """Matrix factorization with neural network and side information."""

    def __init__(
            self, side_info, **kwargs):
        super().__init__(**kwargs)

        self.side_info = side_info

    def __call__(self, x):
        ui, vj = self.features(x)
        s = self.side_info[x[:, 0]]

        x = jnp.concatenate([ui, vj, s], axis=1)
        return self._forward(x)
