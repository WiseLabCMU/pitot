"""Matrix Factorization Models."""

from functools import partial
from jax import numpy as jnp
import haiku as hk

from modules import LearnedFeatures, HybridEmbedding


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

    def __call__(self, *args, full=False, **kwargs):
        """Dispatcher."""
        if full:
            return self._predict_full(*args, **kwargs)
        else:
            return self._predict(*args, **kwargs)


class MatrixFactorization(MFBase):
    """Matrix factorization."""

    def __init__(self, M, D, **kwargs):
        super().__init__(**kwargs)

        self.M = M(samples=self.shape[0], name="M")
        self.D = D(samples=self.shape[1], name="D")

    def _predict(self, ij, ip=None, m_bar=None, d_bar=None):
        """Ordinary Matrix Factorization with External Baseline.

        C_ij_hat = m_bar_i + d_bar_j + m_i^Td_j.

        NOTE: i' (parameter ip) is ignored.
        """
        m_i = self.M(ij[:, 0])
        d_j = self.D(ij[:, 1])
        return (
            m_bar[ij[:, 0]] + d_bar[ij[:, 1]] + jnp.sum(m_i * d_j, axis=1))

    def _predict_full(self, m_bar=None, d_bar=None):
        M = self.M(None)
        D = self.D(None)
        C_hat = (
            m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
            + jnp.matmul(M, D.T))
        return {"C_hat": C_hat, "M": M, "D": D}


class MatrixFactorizationIF(MatrixFactorization):
    """Matrix factorization with interference support."""

    def __init__(self, M, D, K=3, **kwargs):
        super().__init__(M, D, **kwargs)
        self.K = K

    def _predict(self, ij, ip=None, m_bar=None, d_bar=None):
        """Matrix Factorization with Interference.

        C_ij_i'_hat =
            m_bar_i + d_bar_j
            + m_i^Td_j
            + sum_k=1^K (m_i^Tv_s m_i'^Tv_g).
        """
        # Non-interference: m_bar_i + d_bar_j + m_i^Td_j
        m_i = self.M(ij[:, 0])
        d_stack = self.D(ij[:, 1])
        batch, dim = m_i.shape
        d_j = d_stack[:, :dim]
        pred = m_bar[ij[:, 0]] + d_bar[ij[:, 1]] + jnp.sum(m_i * d_j, axis=1)

        if ip is not None:
            # v_s, v_g: [batch, dim, K]
            v_shape = [batch, dim, self.K]
            v_s = d_stack[:, dim:(1 + self.K) * dim].reshape(v_shape)
            v_g = d_stack[(1 + self.K) * dim:].reshape(v_shape)

            # susceptibility, magnitude: [batch, K]
            m_shape = [batch, dim, 1]
            susceptibility = jnp.sum(m_i.reshape(m_shape) * v_s, axis=1)
            magnitude = jnp.sum(self.M(ip).reshape(m_shape) * v_g, axis=1)
            pred += jnp.sum(susceptibility * magnitude, axis=1)

        return pred

    def _predict_full(self, ij=None, ip=None, m_bar=None, d_bar=None):
        # M: [N_m, dim]
        M = self.M(None)
        # D: [N_d, dim]
        d_stack = self.D(None)
        N_m, dim = M.shape[1]
        D = d_stack[:, :dim]
        # C: [N_m, N_d]
        C_hat = (
            m_bar.reshape([-1, 1]) + d_bar.reshape([1, -1])
            + jnp.matmul(M, D.T))

        # v_s, v_g: [N_m, dim, K]
        v_shape = [N_m, dim, self.K]
        v_s = d_stack[:, dim:(1 + self.K) * dim].reshape(v_shape)
        v_g = d_stack[(1 + self.K) * dim:].reshape(v_shape)

        # ij: [samples, 2] -> m_i: [samples, dim]
        m_i = M[ij[:, 0]]
        # v_s_i, v_g_i: [samples, dim, K]
        v_s_i = v_s[ij[:, 1]]
        v_g_i = v_g[ij[:, 1]]
        # ip: [samples] -> d_ip: [samples, dim]
        m_ip = M[ip]

        # susceptibility, magnitude: [samples, K]
        m_shape = [m_i.shape[0], dim, 1]
        susceptibility = jnp.sum(m_i.reshape(m_shape) * v_s_i)
        magnitude = jnp.sum(m_ip.reshape(m_shape) * v_g_i)
        interference = jnp.sum(susceptibility * magnitude, axis=1)

        # C_ij_i'_hat: [samples,]
        C_ij_ip_hat = C_hat[ij[:, 0], ij[:, 1]] + interference

        return {
            "C_hat": C_hat, "M": M, "D": D,
            "v_s": v_s, "v_g": v_g, "C_ij_ip_hat": C_ij_ip_hat}


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
        X_m=None, X_d=None, alpha=0.001, K=3,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Interference."""
    _f = partial(_feature_embedding, dim=dim, scale=scale)
    device_out = layers[-1] * (2 * K + 1)
    return MatrixFactorizationIF(
        _f(X_m, layers=layers), _f(X_d, layers=layers[:-1] + [device_out]),
        K=K, alpha=alpha, shape=shape, name="embedding")
