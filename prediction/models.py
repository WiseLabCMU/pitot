"""Matrix Factorization and Baseline Models.

The exposed models (embedding, interference, linear, naive_mlp, device_mlp)
take a Dataset and kwargs with configuration. The actual classes are not
intended to be used directly.
"""

from functools import partial

import jax
from jax import numpy as jnp
import haiku as hk

from beartype.typing import Optional
from jaxtyping import Integer, Array, PyTree

from .modules import (
    LearnedFeatures, HybridEmbedding, SideInformation, simple_mlp, MultiMLP)
from .rank1 import Rank1, Rank1Solution


MFIndices = Integer[Array, "b f"]
MFBaseline = Optional[Rank1Solution]


class MatrixFactorization(hk.Module):
    """Matrix factorization.

    Parameters
    ----------
    P: Platform embedding `P = f_p(U_p, X_p; W_p)`.
    M: Module embedding `M = f_m(U_m, X_m; W_m)`.
    alpha: Multiplier applied to prediction when predicting baseline residuals.
    log: Whether this model is in log-space or natural space.
    shape: (N_p, N_m) size.
    name: module name.
    """

    def __init__(
        self, P, M, alpha: float = 0.001, log: bool = True,
        shape: tuple[int, int] = (10, 10), name: str = "Matrix Factorization"
    ) -> None:
        super().__init__(name=name)

        self.alpha = alpha
        self.log = log
        self.P = P(samples=shape[0], name="P")
        self.M = M(samples=shape[1], name="M")

    @staticmethod
    def _vvmap(func, x):
        """Apply vmap to input PyTree, repeated for each leaf."""
        return jax.tree_util.tree_map(jax.vmap(func), x)

    def __call__(
        self, ij: MFIndices, baseline: MFBaseline = None, full: bool = False
    ) -> PyTree:
        """Ordinary Matrix Factorization with External Baseline.

        C_ij_hat = C_ij_bar + p_i^Tm_j.

        NOTE: k (column 2) of ij (ijk) is ignored.
        NOTE: ij is a list of arrays.

        Parameters
        ----------
        ij: indices to evaluate. Can be a single (batch, 2) list of pairs, or
            a list of splits to evaluate.
        baseline: baseline to apply. If None, no baseline is used.
        full: if full, also returns additional state information.
        """
        P = self.P(None)
        M = self.M(None)

        if full:
            C_bar = Rank1.predict(baseline)
            if self.log:
                C_hat = C_bar + self.alpha * jnp.matmul(P, M.T)
            else:
                C_hat = jnp.exp(C_bar) * jnp.matmul(P, M.T)

            def _inner(ij):
                i, j = ij[:2]
                return C_hat[i, j]

            C_hat_ij = self._vvmap(_inner, ij)
            return C_hat_ij, {"C_hat": C_hat, "P": P, "M": M}
        else:
            def _inner(ij):
                i, j = ij[:2]
                C_bar = Rank1.predict(baseline, ij[:2])
                if self.log:
                    return C_bar + self.alpha * jnp.dot(P[i], M[j])
                else:
                    return jnp.exp(C_bar) * jnp.dot(P[i], M[j])

            return self._vvmap(_inner, ij)


class MatrixFactorizationIF(MatrixFactorization):
    """Matrix factorization with interference support."""

    def __init__(self, *args, s: int = 3, beta: float = 0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = s
        self.beta = beta

    def __call__(
        self, ijk: MFIndices, baseline: MFBaseline = None, full=False
    ) -> PyTree:
        """Matrix Factorization with Interference.

        C_ijk_hat =
            C_ij_bar + p_i^Tm_j
            + 1[valid k] * sum_t=1^s (m_j^Tv[t]_s m_k^Tv[t]_g).
        """
        M = self.M(None)
        r = M.shape[1]

        pF = self.P(None)
        P = pF[:, :r]
        V_s = self.beta * (pF[:, r:(1 + self.s) * r].reshape([-1, r, self.s]))
        V_g = self.beta * (pF[:, (1 + self.s) * r:].reshape([-1, r, self.s]))

        C_bar = Rank1.predict(baseline)

        def _inner(ijk):
            i, j = ijk[:2]
            mFm = 0.
            for k in ijk[2:]:
                mFm += (k != -1) * jnp.dot(
                    jnp.matmul(V_s[i].T, M[j]), jnp.matmul(V_g[i].T, M[k]))

            return (
                (C_bar[i, j] if baseline else 0.)
                + self.alpha * jnp.dot(P[i], M[j]) + mFm)

        C_hat_ijk = self._vvmap(_inner, ijk)

        if full:
            C_hat = C_bar + self.alpha * jnp.matmul(P, M.T)
            return C_hat_ijk, {
                "C_hat": C_hat, "P": P, "M": M, "V_s": V_s, "V_g": V_g}
        else:
            return C_hat_ijk


class BaselineModel(hk.Module):
    """Abstract class with call wrapper."""

    def __init__(
        self, alpha: float = 0.1, shape: tuple[int, int] = (10, 10),
        name: str = "Baseline"
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.shape = shape

    def _call(self, ij):
        raise NotImplementedError()

    def _lcall(self, ij, baseline: MFBaseline = None):
        def _inner(ij):
            return Rank1.predict(baseline, ij) + self.alpha * self._call(ij)

        return jax.tree_util.tree_map(_inner, ij)

    def __call__(
        self, ij: MFIndices, baseline: MFBaseline = None, full: bool = False
    ) -> PyTree:
        """Non-matrix models."""
        if full:
            x, y = jnp.meshgrid(
                jnp.arange(self.shape[0]), jnp.arange(self.shape[1]),
                indexing='ij')
            mlp = jax.vmap(self._call)(jnp.stack([x, y], axis=-1))
            C_bar = Rank1.predict(baseline)
            C_hat = C_bar + self.alpha * mlp
            return self._lcall(ij, baseline=baseline), {"C_hat": C_hat}
        else:
            return self._lcall(ij, baseline=baseline)


class NaiveMLP(BaselineModel):
    """MLP-only model without matrix embedding."""

    def __init__(
            self, P, M, alpha=0.1, layers=[64, 64], shape=(10, 10),
            name="NaiveMLP"):
        super().__init__(alpha=alpha, shape=shape, name=name)
        self.mlp = simple_mlp(
            list(layers) + [1], activation=jax.nn.tanh, name="mlp")
        self.P = P()
        self.M = M()

    def _call(self, ij):
        x_in = jnp.concatenate([self.P(ij[:, 0]), self.M(ij[:, 1])], axis=1)
        return self.mlp(x_in).reshape(-1)


class LinearCostModel(BaselineModel):
    """Simple model which has a cost for each opcode."""

    def __init__(self, M, shape=(10, 10), name="LinearCostModel"):
        super().__init__(alpha=1.0, shape=shape, name=name)
        self.M = M()
    
    def _call(self, ij):
        # Pad with ones for bias
        opcodes = self.M(ij[:, 1])
        opcodes = jnp.concatenate(
            [jnp.exp(opcodes) - 1, jnp.ones((opcodes.shape[0], 1))], axis=1)
        # Different w for each platform
        # Includes bias (dim already added to opcodes)
        w = hk.get_parameter(
            "w", (opcodes.shape[1],), jnp.float32, init=jnp.zeros)
        speed = hk.get_parameter(
            "speed", (self.shape[0],), jnp.float32, init=jnp.ones)
        return jnp.sum(opcodes * w.reshape(1, -1), axis=1) * speed[ij[:, 0]]


class DeviceModel(BaselineModel):
    """Per-device modeling using WebAssembly as a virtual CPU simulator."""

    def __init__(
            self, M, alpha=0.1, layers=[64, 64], shape=(10, 10),
            name="DeviceModel"):
        super().__init__(alpha=alpha, shape=shape, name=name)
        self.M = M()
        self.mlps = MultiMLP(list(layers) + [1], jax.nn.tanh, shape[0])

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
        dataset, X_p=None, X_m=None, alpha=0.001, log=True,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Side Information using NN Embedding."""
    X_p = dataset.x_p if X_p is True else X_p
    X_m = dataset.x_m if X_m is True else X_m
    _f = partial(_feature_embedding, layers=layers, dim=dim, scale=scale)
    return partial(
        MatrixFactorization,
        _f(X_p), _f(X_m), alpha=alpha, log=log, shape=shape, name="embedding")


def interference(
        dataset, X_p=None, X_m=None, alpha=0.001, s=3, log=True,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Interference."""
    X_p = dataset.x_p if X_p is True else X_p
    X_m = dataset.x_m if X_m is True else X_m
    _f = partial(_feature_embedding, dim=dim, scale=scale)
    device_out = layers[-1] * (2 * s + 1)
    return partial(
        MatrixFactorizationIF,
        _f(X_p, layers=layers[:-1] + [device_out]), _f(X_m, layers=layers),
        s=s, alpha=alpha, log=log, shape=shape, name="embedding")


def linear(_, alpha=0.001, log=True, dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: C_ij = <u_m^{(i)}, u_d^{(j)}>."""
    return partial(
        MatrixFactorization,
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        alpha=alpha, log=log, shape=shape, name="linear")


def naive_mlp(dataset, alpha=0.1, shape=(10, 10), layers=[64, 64]):
    """MLP-only model without matrix embedding."""
    P = partial(SideInformation, dataset.x_p, name="x_p")
    M = partial(SideInformation, dataset.x_m, name="X_m")
    return partial(
        NaiveMLP,
        P, M, layers=layers, alpha=alpha, shape=shape, name="naive_mlp")


def linear_cost(dataset, shape=(10, 10)):
    """Linear opcode cost model."""
    M = partial(SideInformation, dataset.x_m, name="X_m")
    return partial(LinearCostModel, M, shape=shape)


def device_mlp(dataset, alpha=0.1, shape=(10, 10), layers=[64, 64]):
    """Per-device MLP model."""
    M = partial(SideInformation, dataset.x_m, name="X_m")
    return partial(
        DeviceModel,
        M, layers=layers, alpha=alpha, shape=shape, name="device_mlp")
