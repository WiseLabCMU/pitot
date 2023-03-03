"""Matrix Factorization and Baseline Models.

The exposed models (embedding, interference, linear, naive_mlp, device_mlp)
take a Dataset and kwargs with configuration. The actual classes are not
intended to be used directly.
"""

from functools import partial

import jax
from jax import numpy as jnp
import haiku as hk

from beartype.typing import Union, Optional, Callable
from jaxtyping import PyTree, Integer, Array, Float32

from .modules import (
    LearnedFeatures, HybridEmbedding, SideInformation, simple_mlp, MultiMLP)
from .rank1 import Rank1Solution


MFResult = Union[
    Float32[Array, "b"], tuple[Float32[Array, "b"], dict]]
MFIndices = Union[Integer[Array, "b f"], list[Integer[Array, "b f"]]]
MFBaseline = Optional[Rank1Solution]


class MatrixFactorization(hk.Module):
    """Matrix factorization.

    Parameters
    ----------
    P: Platform embedding `P = f_p(U_p, X_p; W_p)`.
    M: Module embedding `M = f_m(U_m, X_m; W_m)`.
    alpha: Multiplier applied to prediction when predicting baseline residuals.
    shape: (N_p, N_m) size.
    name: module name.
    """

    def __init__(
        self, P, M, alpha: float = 0.001,
        shape: tuple[int, int] = (10, 10), name: str = "Matrix Factorization"
    ) -> None:
        super().__init__(name=name)

        self.alpha = alpha
        self.P = P(samples=shape[0], name="P")
        self.M = M(samples=shape[1], name="M")

    @staticmethod
    def _vvmap(func, ij: Union[list, tuple, PyTree]) -> list[PyTree]:
        """Apply vmap to input arguments ij, repeated for each element of ij.

        If ij is not a list or tuple, it is promoted to a list, and vmap is
        applied to each as usual.
        """
        if not isinstance(ij, (list, tuple)):
            return jax.vmap(func)(ij)
        else:
            return [
                None if split is None else jax.vmap(func)(split)
                for split in ij]

    def __call__(
        self, ij: MFIndices, baseline: MFBaseline = None, full: bool = False
    ) -> MFResult:
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
            C_bar = Rank1Solution.predict(baseline) if baseline else 0
            C_hat = C_bar + self.alpha * jnp.matmul(P, M.T)

            def _inner(ij):
                i, j = ij[:2]
                return C_hat[i, j]

            C_hat_ij = self._vvmap(_inner, ij)
            return C_hat_ij, {"C_hat": C_hat, "P": P, "M": M}
        else:
            def _inner(ij):
                i, j = ij[:2]
                C_bar = (
                    Rank1Solution.predict(baseline, ij[:2]) if baseline else 0)
                return C_bar + self.alpha * jnp.dot(P[i], M[j])

            return self._vvmap(_inner, ij)


class MatrixFactorizationIF(MatrixFactorization):
    """Matrix factorization with interference support.

    TODO: convert this to new format.
    """

    def __init__(self, *args, s: int = 3, beta: float = 0.001, **kwargs):
        raise NotImplementedError()
        super().__init__(*args, **kwargs)
        self.s = s
        self.beta = beta

    def __call__(
        self, ijk: MFIndices, baseline: MFBaseline = None, full=False
    ) -> MFResult:
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

        C_bar = (
            baseline.predict() if baseline
            else jnp.zeros(P.shape[0], M.shape[0]))

        def _inner(ijk):
            i, j = ijk[:2]
            mFm = 0.
            for k in ijk[2:]:
                mFm += (k != -1) * jnp.dot(
                    jnp.matmul(V_s[i].T, M[j]), jnp.matmul(V_g[i].T, M[k]))
            return C_bar[i, j] + self.alpha * jnp.dot(P[i], M[j]) + mFm

        C_hat_ijk = self._vvmap(_inner, ijk)

        if full:
            C_hat = C_bar + self.alpha * jnp.matmul(P, M.T)
            return C_hat_ijk, {
                "C_hat": C_hat, "P": P, "M": M, "V_s": V_s, "V_g": V_g}
        else:
            return C_hat_ijk


class BaselineModel(hk.Module):
    """Abstract class with call wrapper."""

    def _call(self, ij):
        raise NotImplementedError()

    def _lcall(self, ij, baseline: MFBaseline = None):
        if not isinstance(ij, (list, tuple)):
            ij = [ij]
        return [
            baseline.predict(split) + self.alpha * self._call(split)
            for split in ij]

    def __call__(
        self, ij: MFIndices, baseline: MFBaseline = False, full: bool = False
    ) -> MFResult:
        """Non-matrix models."""
        if full:
            x, y = jnp.meshgrid(
                jnp.arange(self.shape[0]), jnp.arange(self.shape[1]))
            mlp = jax.vmap(self._call)(jnp.stack([x, y], axis=-1)).T
            C_bar = baseline.predict() if baseline else 0
            C_hat = C_bar + self.alpha * mlp
            return self._lcall(ij, baseline=baseline), {"C_hat": C_hat}
        else:
            return self._lcall(ij, baseline=baseline)


class NaiveMLP(BaselineModel):
    """MLP-only model without matrix embedding."""

    def __init__(
            self, P, M, alpha=0.1, layers=[64, 64], shape=(10, 10),
            name="NaiveMLP"):
        super().__init__(name=name)
        self.mlp = simple_mlp(
            list(layers) + [1], activation=jax.nn.tanh, name="mlp")
        self.P = P
        self.M = M
        self.alpha = alpha
        self.shape = shape

    def _call(self, ij):
        x_in = jnp.concatenate([self.P(ij[:, 0]), self.M(ij[:, 1])], axis=1)
        return self.mlp(x_in).reshape(-1)


class DeviceModel(BaselineModel):
    """Per-device modeling using WebAssembly as a virtual CPU simulator."""

    def __init__(
            self, M, alpha=0.1, layers=[64, 64], shape=(10, 10),
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
        dataset, X_p=None, X_m=None, alpha=0.001,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Side Information using NN Embedding."""
    X_p = dataset.x_p if X_p is True else X_p
    X_m = dataset.x_m if X_m is True else X_m
    _f = partial(_feature_embedding, layers=layers, dim=dim, scale=scale)
    return MatrixFactorization(
        _f(X_p), _f(X_m), alpha=alpha, shape=shape, name="embedding")


def interference(
        dataset, X_p=None, X_m=None, alpha=0.001, s=3,
        shape=(10, 10), layers=[64, 32], dim=4, scale=0.01):
    """Matrix Factorization with Interference."""
    X_p = dataset.x_p if X_p is True else X_p
    X_m = dataset.x_m if X_m is True else X_m
    _f = partial(_feature_embedding, dim=dim, scale=scale)
    device_out = layers[-1] * (2 * s + 1)
    return MatrixFactorizationIF(
        _f(X_p, layers=layers[:-1] + [device_out]), _f(X_m, layers=layers),
        s=s, alpha=alpha, shape=shape, name="embedding")


def linear(_, alpha=0.001, dim=32, shape=(10, 10), scale=0.01):
    """Linear matrix factorization: C_ij = <u_m^{(i)}, u_d^{(j)}>."""
    return MatrixFactorization(
        partial(LearnedFeatures, dim=dim, scale=scale),
        partial(LearnedFeatures, dim=dim, scale=scale),
        alpha=alpha, shape=shape, name="linear")


def naive_mlp(dataset, alpha=0.1, shape=(10, 10), layers=[64, 64]):
    """MLP-only model without matrix embedding."""
    P = SideInformation(dataset.x_p, name="x_p")
    M = SideInformation(dataset.x_m, name="X_m")
    return NaiveMLP(P, M, layers=layers, shape=shape, name="naive_mlp")


def device_mlp(dataset, alpha=0.1, shape=(10, 10), layers=[64, 64]):
    """Per-device MLP model."""
    M = SideInformation(dataset.x_m, name="x_m")
    return DeviceModel(M, layers=layers, shape=shape, name="device_mlp")
