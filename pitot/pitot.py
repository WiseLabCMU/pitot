"""Pitot model."""

import haiku as hk
import optax
import jax
from jax import numpy as jnp
from jax import random

from jaxtyping import PyTree
from beartype.typing import Optional, Callable, cast

from prediction import (
    MatrixCompletionModel, embeddings, loss, ObjectiveSet, types, Split)
from prediction.utils import array_unpack

from .linear_scaling import LinearScaling


class Pitot(MatrixCompletionModel):
    """Pitot: interference-aware matrix factorization with side information.

    Parameters
    ----------
    mf_dim: Matrix factorization embedding dimension.
    if_dim: Interference dimension, i.e. the number of types of interference.
    if_slope: Rectified interference leaky ReLU slope.
    do_baseline: Compute linear scaling baseline, and learn the residual.
    """

    def __init__(
        self, objectives: ObjectiveSet, losses: loss.Loss,
        workload: Callable[..., embeddings.BaseEmbedding],
        platform: Callable[..., embeddings.BaseEmbedding],
        optimizer: Optional[optax.GradientTransformation] = None,
        mf_dim: int = 128, if_dim: int = 4, if_slope: float = 0.1,
        do_baseline: bool = True
    ) -> None:
        self.mf_dim = mf_dim
        self.if_dim = if_dim
        self.if_slope = if_slope
        self.do_baseline = do_baseline

        def _transform(f, dim: int = 0):
            def forward(*args, **kwargs):
                return f(dim=dim)(*args, **kwargs)
            return hk.without_apply_rng(hk.transform(forward))

        # Workload embedding: N full dimensions
        self.f_workload = _transform(workload, dim=mf_dim * losses.N)

        # Platform embedding: 1 full dimension main embedding
        #   + `if_dim * 2` full dimension interference embeddings
        self.f_platform = _transform(
            platform, dim=mf_dim + if_dim * mf_dim * 2)

        super().__init__(
            losses=losses, objectives=objectives, optimizer=optimizer,
            name="Pitot")

    def _init(
        self, key: random.PRNGKeyArray, splits: dict[str, Split]
    ) -> PyTree:
        """Get model parameters."""
        k1, k2 = random.split(key, 2)
        if self.do_baseline:
            data = self.objectives['mf'].index(
                jnp.concatenate([splits['mf'].train, splits['mf'].val]))
            baseline = LinearScaling(
                shape=cast(tuple[int, int], self.objectives["mf"].shape),
                init_val=0., max_iter=10000, tol=1e-5
            ).fit(data)
        else:
            baseline = None

        return {
            "platform": self.f_platform.init(k1, jnp.arange(2)),
            "workload": self.f_workload.init(k2, jnp.arange(2)),
            "baseline": baseline}

    def evaluate(
        self, params: PyTree, data: dict[str, types.Data]
    ) -> dict[str, types.Predictions]:
        """Evaluate model predictions.

        Notes
        -----
        - `X`: related to the actual primary embedding.
        - `w`, `p`: related to workloads and platforms, respectively
        - `V` (`Vs`, `Vg`): related to the interference embedding, split
            between susceptibility (`Vs`) and magnitude (`Vg`).
        - `F`: the interference matrix; not explicitly computed.

        Parameters
        ----------
        params: parameters for all embedding models.
        data: input indices (x) and actual values (y), organized by dataset.

        Returns
        -------
        Batch predictions for each loss.
        """
        # Pre-compute embeddings since Nw, Np << batch.
        Xw, = array_unpack(
            self.f_workload.apply(params["workload"], None),
            (self.loss_func.N, self.mf_dim))

        Xp, Vs, Vg = array_unpack(
            self.f_platform.apply(params["platform"], None),
            (self.mf_dim,),
            (self.if_dim, self.mf_dim),
            (self.if_dim, self.mf_dim))

        def _evaluate(xy: types.Data):
            iw = xy.x['workload']
            ip = xy.x['platform']

            C_bar = LinearScaling.evaluate(params["baseline"], xy)

            # Primary prediction
            XpXw = jnp.sum(Xw[iw, :, :] * Xp[ip, None, :], axis=-1)

            # Interference term: susceptibility, magnitude
            S = jnp.sum(Xw[iw, :, None, :] * Vs[ip, None, :, :], axis=-1)
            M = jnp.zeros((self.loss_func.N))[None, :, None]
            for k in xy.x:
                if k.startswith('interference'):
                    M += jnp.sum(
                        Xw[iw, :, None, :] * Vg[xy.x[k], None, :, :], axis=-1)       
            if self.if_slope == 1.0:
                # Slope = 1  -->  same as no activation. Skip to be safe.
                M_rect = M
            else:
                M_rect = jax.nn.leaky_relu(M, negative_slope=self.if_slope)
            XwFXp = jnp.sum(S * M_rect, axis=-1)

            y_hat = C_bar + XpXw + XwFXp
            return types.Predictions(y_true=xy.y, y_hat=y_hat)

        return {k: _evaluate(v) for k, v in data.items()}

    @classmethod
    def from_config(
        cls, objectives: ObjectiveSet,
        platform_embedding: str = "HybridEmbedding", platform_args: dict = {},
        workload_embedding: str = "HybridEmbedding", workload_args: dict = {},
        loss_class: str = "Squared", loss_args: dict = {},
        optimizer: str = "adamax", optimizer_args: dict = {}, **kwargs
    ) -> "Pitot":
        """Create from configuration."""
        platform = getattr(embeddings, platform_embedding).from_config(
            features=objectives["mf"].data["platform"],
            name="Platform", **platform_args)
        workload = getattr(embeddings, workload_embedding).from_config(
            features=objectives["mf"].data["workload"],
            name="Workload", **workload_args)
        loss_func = getattr(loss, loss_class).from_config(**loss_args)
        opt = getattr(optax, optimizer)(**optimizer_args)
        return cls(
            objectives, loss_func, workload=workload, platform=platform,
            optimizer=opt, **kwargs)
