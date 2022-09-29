"""Matrix Factorization Models.

The exposed models (embedding, interference, linear) take a Dataset and
kwargs with configuration.
"""

from functools import partial
from jax import numpy as jnp
from jax import vmap
import haiku as hk

from .modules import LearnedFeatures, HybridEmbedding


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
        if not isinstance(ij, (list, tuple)):
            ij = [ij]
        return [None if split is None else vmap(func)(split) for split in ij]

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

    def __init__(self, *args, s=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s

    def __call__(self, ijk, m_bar=None, d_bar=None, full=False):
        """Matrix Factorization with Interference.

        C_ijk =
            m_bar_i + d_bar_j + m_i^Td_j
            + 1[valid k] * sum_k=1^K (m_i^Tv_s m_i'^Tv_g).
        """
        M = self.M(None)
        d_stack = self.D(None)
        D = d_stack[:, :M.shape[1]]

        r = M.shape[1]
        V_s = d_stack[:, r:(1 + self.s) * r].reshape([-1, r, self.s])
        V_g = d_stack[:, (1 + self.s) * r:].reshape([-1, r, self.s])

        def _inner(ijk):
            i, j, k = ijk
            mFm = (k != -1) * jnp.dot(
                jnp.matmul(V_s[j].T, M[i]), jnp.matmul(V_g[j].T, M[k]))
            return (
                m_bar[i] + d_bar[j] + self.alpha * (jnp.dot(M[i], D[j]) + mFm))

        if full:
            C_hat = (
                m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
                + self.alpha * jnp.matmul(M, D.T))
            C_hat_ijk = self._vvmap(_inner, ijk)
            return C_hat_ijk, {
                "C_hat": C_hat, "M": M, "D": D, "V_s": V_s, "V_g": V_g}
        else:
            return self._vvmap(_inner, ijk)


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
