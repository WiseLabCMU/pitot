"""Matrix Factorization and Baseline Models.

The exposed models (embedding, interference, linear, naive_mlp, device_mlp)
take a Dataset and kwargs with configuration. The actual classes are not
intended to be used directly.
"""

from functools import partial

import jax
from jax import numpy as jnp
import haiku as hk

from .modules import (
    LearnedFeatures, HybridEmbedding, SideInformation, simple_mlp, MultiMLP)


class MatrixFactorization(hk.Module):
    """Matrix factorization.

    Parameters
    ----------
    M : hk.Module
        Module embedding M = f_m(U_m, X_m; W_m).
    D : hk.Module
        Device embedding D = f_d(U_d, X_d; W_d).
    alpha : float
        Multiplier applied to prediction when predicting baseline residuals.
    shape : (int, int)
        (N_m, N_d) size.
    """

    def __init__(
            self, M, D, alpha=0.001, shape=(10, 10),
            name="Matrix Factorization"):
        super().__init__(name=name)

        self.alpha = alpha
        self.M = M(samples=shape[0], name="M")
        self.D = D(samples=shape[1], name="D")

    @staticmethod
    def _vvmap(func, ij):
        """Apply vmap to input arguments ij, repeated for each element of ij.

        If ij is not a list or tuple, it is promoted to a list, and vmap is
        applied to each as usual.
        """
        if not isinstance(ij, (list, tuple)):
            ij = [ij]
        return [
            None if split is None else jax.vmap(func)(split)
            for split in ij]

    def __call__(self, ij, m_bar=None, d_bar=None, full=False):
        """Ordinary Matrix Factorization with External Baseline.

        C_ij_hat = m_bar_i + d_bar_j + m_i^Td_j.

        NOTE: k (column 2) of ij (ijk) is ignored.
        NOTE: ij is a list of arrays.
        """
        M = self.M(None)
        D = self.D(None)

        if full:
            C_hat = (
                m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
                + self.alpha * jnp.matmul(M, D.T))

            def _inner(ij):
                i, j = ij[:2]
                return C_hat[i, j]

            C_hat_ij = self._vvmap(_inner, ij)
            return C_hat_ij, {"C_hat": C_hat, "M": M, "D": D}
        else:
            def _inner(ij):
                i, j = ij[:2]
                return m_bar[i] + d_bar[j] + self.alpha * jnp.dot(M[i], D[j])

            return self._vvmap(_inner, ij)


class MatrixFactorizationIF(MatrixFactorization):
    """Matrix factorization with interference support."""

    def __init__(self, *args, s=3, beta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s
        self.beta = beta

    def __call__(self, ijk, m_bar=None, d_bar=None, full=False):
        """Matrix Factorization with Interference.

        C_ijk =
            m_bar_i + d_bar_j + m_i^Td_j
            + 1[valid k] * sum_t=1^s (m_i^Tv[t]_s m_k^Tv[t]_g).
        """
        M = self.M(None)
        r = M.shape[1]
        dF = self.D(None)
        D = dF[:, :r]

        V_s = self.beta * (dF[:, r:(1 + self.s) * r].reshape([-1, r, self.s]))
        V_g = self.beta * (dF[:, (1 + self.s) * r:].reshape([-1, r, self.s]))

        def _inner(ijk):
            i, j = ijk[:2]
            mFm = 0.
            for k in ijk[2:]:
                mFm += (k != -1) * jnp.dot(
                    jnp.matmul(V_s[j].T, M[i]), jnp.matmul(V_g[j].T, M[k]))
            return (
                m_bar[i] + d_bar[j] + self.alpha * jnp.dot(M[i], D[j]) + mFm)

        C_hat_ijk = self._vvmap(_inner, ijk)

        if full:
            C_hat = (
                m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
                + self.alpha * jnp.matmul(M, D.T))
            return C_hat_ijk, {
                "C_hat": C_hat, "M": M, "D": D, "V_s": V_s, "V_g": V_g}
        else:
            return C_hat_ijk


class BaselineModel(hk.Module):
    """Abstract class with call wrapper."""

    def _call(self, ij):
        raise NotImplementedError()

    def _lcall(self, ij):
        if not isinstance(ij, (list, tuple)):
            ij = [ij]
        return [self._call(split) for split in ij]

    def __call__(self, ij, m_bar=None, d_bar=None, full=False):
        """Non-matrix models."""
        baseline = m_bar[ij[:, 0]] + d_bar[ij[:, 1]]

        if full:
            x, y = jnp.meshgrid(
                jnp.arange(self.shape[0]), jnp.arange(self.shape[1]))
            mlp = jax.vmap(self._call)(jnp.stack([x, y], axis=-1)).T
            C_hat = (
                m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
                + self.alpha * mlp)
            return baseline + self.alpha * self._lcall(ij), {"C_hat": C_hat}
        else:
            return baseline + self.alpha * self._lcall(ij)


class NaiveMLP(BaselineModel):
    """MLP-only model without matrix embedding."""

    def __init__(
            self, M, D, alpha=0.0001, layers=[64, 64], shape=(10, 10),
            name="NaiveMLP"):
        super().__init__(name=name)
        self.mlp = simple_mlp(
            list(layers) + [1], activation=jax.nn.tanh, name="mlp")
        self.M = M
        self.D = D
        self.alpha = alpha
        self.shape = shape

    def _call(self, ij):
        x_in = jnp.concatenate([self.M(ij[:, 0]), self.D(ij[:, 1])], axis=1)
        return self.mlp(x_in).reshape(-1)


class DeviceModel(BaselineModel):
    """Per-device modeling using WebAssembly as a virtual CPU simulator."""

    def __init__(
            self, M, alpha=0.0001, layers=[64, 64], shape=(10, 10),
            name="DeviceModel"):
        super().__init__(name=name)
        self.M = M
        self.mlps = MultiMLP(list(layers) + [1], jax.nn.tanh, shape[1])
        self.alpha = alpha
        self.shape = shape

    def _call(self, ij):
        res = self.mlps(self.M(ij[:, 0]), ij[:, 1]).reshape(-1)
        return res


def _feature_embedding(data, layers=[], dim=4, scale=0.01):
    """Create feature embedding."""
    if data is None:
        return partial(LearnedFeatures, dim=layers[-1], scale=scale)
    else:
        return partial(
            HybridEmbedding, data, layers=layers, dim=dim, scale=scale)


def embedding(
        dataset, X_m=None, X_d=None, alpha=0.001,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Side Information using NN Embedding."""
    X_m = dataset.x_m if X_m is True else X_m
    X_d = dataset.x_d if X_d is True else X_d
    _f = partial(_feature_embedding, layers=layers, dim=dim, scale=scale)
    return MatrixFactorization(
        _f(X_m), _f(X_d), alpha=alpha, shape=shape, name="embedding")


def interference(
        dataset, X_m=None, X_d=None, alpha=0.001, s=3,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Interference."""
    X_m = dataset.x_m if X_m is True else X_m
    X_d = dataset.x_d if X_d is True else X_d
    _f = partial(_feature_embedding, dim=dim, scale=scale)
    device_out = layers[-1] * (2 * s + 1)
    return MatrixFactorizationIF(
        _f(X_m, layers=layers), _f(X_d, layers=layers[:-1] + [device_out]),
        s=s, alpha=alpha, shape=shape, name="embedding")


def linear(_, alpha=0.001, dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: C_ij = <u_m^{(i)}, u_d^{(j)}>."""
    return MatrixFactorization(
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        alpha=alpha, shape=shape, name="linear")


def naive_mlp(dataset, alpha=0.001, shape=(10, 10), layers=[64, 64]):
    """MLP-only model without matrix embedding."""
    M = SideInformation(dataset.x_m, name="x_m")
    D = SideInformation(dataset.x_d, name="X_d")
    return NaiveMLP(M, D, layers=layers, shape=shape, name="naive_mlp")


def device_mlp(dataset, alpha=0.001, shape=(10, 10), layers=[64, 64]):
    """Per-device MLP model."""
    M = SideInformation(dataset.x_m, name="x_m")
    return DeviceModel(M, layers=layers, shape=shape, name="device_mlp")
