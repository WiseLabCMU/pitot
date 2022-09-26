"""Matrix Factorization Models."""

from functools import partial
from jax import numpy as jnp
import haiku as hk

from .modules import LearnedFeatures, HybridEmbedding


class MFBase(hk.Module):
    """Matrix Factorization base class.

    Parameters
    ----------
    alpha : float
        Multiplier applied to prediction when predicting baseline residuals.
    shape : (int, int)
        (N_m, N_d) size.
    """

    def __init__(
            self, alpha=0.001, shape=(10, 10), name="Matrix Factorization"):
        super().__init__(name=name)
        self.shape = shape
        self.alpha = alpha

    def __call__(self, ij, ip=None, baseline=None):
        """Dispatcher.

        Parameters
        ----------
        ij : jnp.array(int[:, 2]) or None
            If not None, generates predictions for the given indices;
            otherwise, generate predicted full non-interference matrix.
        ip : jnp.array(int[])
            Optional interference indices.
        baseline : jnp.array(float[N_m, N_d])
            Baseline (non-interference, rank 1) execution time; is added to
            the predictions.

        Returns
        -------
        jnp.array(float[:]) or full results
            Predicted execution time C_hat_ij for each ``ij`` pair
            (with optional ``ip`` interferer). If ``ij = None``, then returns
            full results with a (C_hat, m, d) triple.
        """
        if ij is None:
            pred = self._predict_full()
            if baseline is not None:
                pred = pred * self.alpha + baseline
        else:
            pred = self._predict(ij, ip=ip)
            if baseline is not None:
                pred = pred * self.alpha + baseline[ij[:, 0], ij[:, 1]]
        return pred

    def _predict_full(self):
        """Generate full matrix predictions.

        Embeddings are not available by default; specific implementations must
        specify how to compute them.
        """
        x, y = jnp.meshgrid(
            jnp.arange(self.shape[0]), jnp.arange(self.shape[1]))
        coords = jnp.stack([x.reshape(-1), y.reshape(-1)]).T
        pred = self._predict(coords)
        return (
            jnp.zeros(self.shape).at[coords[:, 0], coords[:, 1]].set(pred),
            0.0, 0.0)


class MatrixFactorization(MFBase):
    """Matrix factorization."""

    def __init__(self, M, D, **kwargs):
        super().__init__(**kwargs)

        self.M = M(samples=self.shape[0], name="M")
        self.D = D(samples=self.shape[1], name="D")

    def _predict(self, ij, ip=None):
        m_i = self.M(ij[:, 0])
        d_j = self.D(ij[:, 1])
        return jnp.sum(m_i * d_j, axis=1)

    def _predict_full(self):
        M = self.M(None)
        D = self.D(None)
        return jnp.matmul(M, D.T), M, D


class MatrixFactorizationIF(MatrixFactorization):
    """Matrix factorization with interference support."""

    def __init__(self, M, D, **kwargs):
        super().__init__(M, D, **kwargs)

    def _predict(self, ij, ip=None):
        m_i = self.M(ij[:, 0])
        d_stack = self.D(ij[:, 1])
        dim = m_i.shape[-1]
        d_j = d_stack[:, :dim]

        pred = jnp.sum(m_i * d_j, axis=1)

        if ip is not None:
            v_s = d_stack[:, dim:2 * dim]
            v_g = d_stack[2 * dim:]

            susceptibility = jnp.sum(m_i * v_s, axis=1)
            magnitude = jnp.sum(self.M(ip) * v_g, axis=1)
            pred += jnp.sum(susceptibility * magnitude)

        return pred


def linear(alpha=0.001, dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: C_ij = <u_m^{(i)}, u_d^{(j)}>."""
    return MatrixFactorization(
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        alpha=alpha, shape=shape, name="linear")


def _feature_embedding(data, layers=[], dim=4, scale=0.01):
    """Create feature embedding."""
    if data is None:
        return partial(LearnedFeatures, dim=layers[-1], scale=scale)
    else:
        return partial(
            HybridEmbedding, data, layers=layers, dim=dim, scale=scale)


def embedding(
        X_m=None, X_d=None, alpha=0.001,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Side Information using NN Embedding."""
    _f = partial(_feature_embedding, layers=layers, dim=dim, scale=scale)
    return MatrixFactorization(
        _f(X_m), _f(X_d), alpha=alpha, shape=shape, name="embedding")


def interference(
        X_m=None, X_d=None, alpha=0.001,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Interference."""
    _f = partial(_feature_embedding, dim=dim, scale=scale)
    return MatrixFactorizationIF(
        _f(X_m, layers=layers), _f(X_d, layers=layers[:-1] + [layers[-1] * 3]),
        alpha=alpha, shape=shape, name="embedding")
