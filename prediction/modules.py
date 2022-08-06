"""Matrix factorization components."""

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
