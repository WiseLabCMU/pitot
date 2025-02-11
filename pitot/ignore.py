"""Model ignoring interference effects."""

import haiku as hk
import optax
from beartype.typing import Callable, Optional
from jax import numpy as jnp
from jaxtyping import PyTree

from prediction import MatrixCompletionModel, ObjectiveSet, embeddings, loss, types

from .linear_scaling import LinearScaling
from .pitot import Pitot


class PitotIgnore(Pitot):
    """Non-interference aware matrix completion.

    Parameters
    ----------
    mf_dim: Matrix factorization embedding dimension.
    do_baseline: Compute linear scaling baseline, and learn the residual.
    """

    def __init__(
        self, objectives: ObjectiveSet, losses: loss.Loss,
        workload: Callable[..., embeddings.BaseEmbedding],
        platform: Callable[..., embeddings.BaseEmbedding],
        optimizer: Optional[optax.GradientTransformation] = None,
        mf_dim: int = 128,
        do_baseline: bool = True
    ) -> None:
        self.mf_dim = mf_dim
        self.do_baseline = do_baseline

        def _transform(f, dim: int = 0):
            def forward(*args, **kwargs):
                return f(dim=dim)(*args, **kwargs)
            return hk.without_apply_rng(hk.transform(forward))

        self.f_workload = _transform(workload, dim=mf_dim * losses.N)
        self.f_platform = _transform(platform, dim=mf_dim)

        MatrixCompletionModel.__init__(
            self, losses=losses, objectives=objectives, optimizer=optimizer,
            name="PitotIgnore")

    def evaluate(
        self, params: PyTree, data: dict[str, types.Data]
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
        Xw = self.f_workload.apply(
            params["workload"], None
        ).reshape(-1, self.loss_func.N, self.mf_dim)
        Xp = self.f_platform.apply(params["platform"], None)

        def _evaluate(xy: types.Data):
            iw = xy.x['workload']
            ip = xy.x['platform']

            C_bar = LinearScaling.evaluate(params["baseline"], xy)
            XpXw = jnp.sum(Xw[iw, :, :] * Xp[ip, None, :], axis=-1)
            y_hat = C_bar + XpXw
            return types.Predictions(y_true=xy.y, y_hat=y_hat)

        return {k: _evaluate(v) for k, v in data.items()}
