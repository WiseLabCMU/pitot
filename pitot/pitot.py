"""Pitot model."""

import haiku as hk
import optax
import jax
from jax import numpy as jnp
from jax import random

from jaxtyping import PyTree
from beartype.typing import Optional, Any, Callable

from prediction import (
    MatrixCompletionModel, embeddings, loss, ObjectiveSet, types)
from prediction.utils import array_unpack


class Pitot(MatrixCompletionModel):
    """Pitot: interference-aware matrix factorization with side information."""

    def __init__(
        self, workload: Callable[..., embeddings.BaseEmbedding],
        platform: Callable[..., embeddings.BaseEmbedding], losses: loss.Loss,
        objectives: ObjectiveSet, baseline: Optional[Any] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
        mf_dim: int = 128, if_dim: int = 4, qr_dim: int = 16,
        if_slope: float = 0.1
    ) -> None:
        self.mf_dim = mf_dim
        self.if_dim = if_dim
        self.qr_dim = qr_dim
        self.baseline = baseline
        self.if_slope = if_slope

        def _transform(f, dim: int = 0):
            def forward(*args, **kwargs):
                return f(dim=dim)(*args, **kwargs)
            return hk.without_apply_rng(hk.transform(forward))

        # Workload embedding: 1 full dimension main embedding + (N-1) quantile
        # offset low rank additions
        self.f_workload = _transform(
            workload, dim=mf_dim + qr_dim * (losses.N - 1))

        # Platform embedding: 1 full dimension main embedding
        #   + `(quantiles - 1)` QR offset low rank additions
        #   + `if_dim * 2` full dimension interference embeddings
        #   + `qr_dim * (quantiles - 1) * 2` QR offset additions
        self.f_platform = _transform(
            platform, dim=(
                mf_dim
                + qr_dim * (losses.N - 1)
                + if_dim * mf_dim * 2
                + if_dim * qr_dim * (losses.N - 1) * 2))

        super().__init__(
            losses=losses, objectives=objectives, optimizer=optimizer,
            name="Pitot")

    def _init(self, key: random.PRNGKeyArray) -> PyTree:
        """Get model parameters."""
        k1, k2 = random.split(key, 2)
        return {
            "platform": self.f_platform.init(k1, jnp.arange(2)),
            "workload": self.f_workload.init(k2, jnp.arange(2))
        }

    def evaluate(
        self, params: PyTree, data: dict[str, types.Data],
    ) -> dict[str, types.Predictions]:
        """Evaluate model predictions.

        Parameters
        ----------
        params: parameters for all embedding models.
        data: input indices (x) and actual values (y), organized by dataset.

        Returns
        -------
        Batch predictions for each loss.
        """
        def _evaluate(xy: types.Data):
            _nq = self.loss_func.N - 1
            Ew = self.f_workload.apply(params["workload"], None)
            Ew, Xw  = array_unpack(Ew, (self.mf_dim,))
            Ew, XQw = array_unpack(Ew, (_nq, self.qr_dim))
            assert Ew.shape[1] == 0

            Ep = self.f_platform.apply(params["platform"], None)
            Ep, Xp  = array_unpack(Ep, (self.mf_dim,))
            Ep, XQp = array_unpack(Ep, (_nq, self.qr_dim))
            Ep, Vs  = array_unpack(Ep, (self.if_dim, self.mf_dim))
            Ep, Vg  = array_unpack(Ep, (self.if_dim, self.mf_dim))
            Ep, VQs = array_unpack(Ep, (_nq, self.if_dim, self.qr_dim))
            Ep, VQg = array_unpack(Ep, (_nq, self.if_dim, self.qr_dim))
            assert Ep.shape[1] == 0

            C_bar = 0.0   # LinearScaling.predict(self.baseline)

            XpXw = jnp.sum(
                Xw[xy.x['workload']] * Xp[xy.x['platform']], axis=-1)

            susceptibility = jnp.sum(
                Xw[xy.x['workload'], None, :] * Vs[xy.x['platform']], axis=-1)
            magnitude = 0.
            for k in xy.x:
                if k.startswith('interference'):
                    magnitude += jnp.sum(
                        Xw[xy.x['workload'], None, :] * Vg[xy.x[k]], axis=-1)

            rectified_magnitude = jax.nn.leaky_relu(
                magnitude, negative_slope=self.if_slope)
            mFm = jnp.sum(susceptibility * rectified_magnitude, axis=-1)

            # TODO: add QR
            return types.Predictions(
                y_true=xy.y, y_hat=(C_bar + XpXw + mFm)[:, None])

        return {k: _evaluate(v) for k, v in data.items()}
