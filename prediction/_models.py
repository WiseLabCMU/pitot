

class MFNN(MFBase):
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
            self, side_info=None, **kwargs):
        super().__init__(**kwargs)

        self.side_info = side_info

    def __call__(self, x):
        ui, vj = self.features(x)
        s = self.side_info[x[:, 0]]

        x = jnp.concatenate([ui, vj, s], axis=1)
        return self._forward(x)


class MFNNResidual(MFNN):
    """Matrix factorization with residual after rank 1 factorization."""

    def __init__(
            self, side_info=None, **kwargs):
        super().__init__(**kwargs)

        self.side_info = side_info
        self.baseline = MFLinear(dim=1, samples=kwargs["samples"])

    def __call__(self, x):
        ui, vj = self.features(x)
        s = self.side_info[x[:, 0]]

        x = jnp.concatenate([ui, vj, s], axis=1)
        return self._forward(x) + self.baseline(x)


class LinearModel(MFBase):
    """Matrix factorization as a linear model."""

    def __init__(self, features=None, scale=0.01, samples=(10, 10), name=None):
        super().__init__(name=name)
        self.features = features
        self.samples = samples
        self.scale = scale

    def __call__(self, x):
        init = hk.initializers.RandomUniform(minval=0, maxval=self.scale)

        V = hk.get_parameter(
            "V", shape=(self.samples[1], self.features.shape[1]), init=init)
        return jnp.log(jnp.sum(jnp.exp(
            self.features[x[:, 0]] + V[x[:, 1]]), axis=1))


class Embedding(MFBase):
    """Feature embedding approach."""

    optimizer = optax.adam(0.001)
    epochs = 100
    epoch_size = 500

    def __init__(self, side_info=None, scale=0.01, samples=(10, 10), name=None):
        super().__init__(name=name)
        self.side_info = side_info
        self.samples = samples

        self.features = FeatureLookup(
            dim=(4, 32), samples=samples, scale=scale)

        self.tower_i = hk.Sequential([
            hk.Linear(64), jax.nn.sigmoid,
            hk.Linear(32), jax.nn.sigmoid,
            hk.Linear(32)
        ], name='tower_i')

    def __call__(self, x):
        ui, vj = self.features(x)
        s = self.side_info[x[:, 0]]

        proj_i = self.tower_i(jnp.concatenate([ui, s], axis=1))

        return jnp.log(jnp.sum(jnp.exp(proj_i + vj), axis=1))
