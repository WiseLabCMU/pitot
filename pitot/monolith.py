"""Monolithic MLP."""

import haiku as hk
import jax
import optax
from beartype.typing import Optional
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from prediction import MatrixCompletionModel, ObjectiveSet, Split, loss, types


def _mlp(layers: list[int], dim: int) -> hk.Module:
    _layers = sum([
        [hk.Linear(d, name="mlp_{}".format(i)), jax.nn.gelu]  # type: ignore
        for i, d in enumerate(layers)
    ], []) + [hk.Linear(dim, name="mlp_out")]                 # type: ignore
    return hk.Sequential(_layers, name="mlp")                 # type: ignore


class Monolith(MatrixCompletionModel):
    """Monolithic matrix completion model from side information.

    Parameters
    ----------
    layers: neural network layers.
    """

    def __init__(
        self, objectives: ObjectiveSet, losses: loss.Loss,
        optimizer: Optional[optax.GradientTransformation] = None,
        layers: list[int] = [256, 256]
    ) -> None:
        def forward(*args, **kwargs):
            return _mlp(layers=layers, dim=losses.N)(*args, **kwargs)

        self.nn = hk.without_apply_rng(hk.transform(forward))

        super().__init__(
            objectives=objectives, losses=losses, optimizer=optimizer,
            name="Monolith")

    def _init(
        self, key: PRNGKeyArray, splits: dict[str, Split]
    ) -> PyTree:
        """Get model parameters."""
        mf_dim = (
            self.objectives['mf'].data['platform'].shape[1]
            + self.objectives['mf'].data['workload'].shape[1])
        if_dim = (
            self.objectives['mf'].data['platform'].shape[1]
            + 2 * self.objectives['mf'].data['workload'].shape[1])
        return {
            "mf": self.nn.init(key, jnp.zeros((2, mf_dim))),
            "if": self.nn.init(key, jnp.zeros((2, if_dim)))}

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
        def _evaluate(xy: types.Data):
            ip = xy.x['platform']
            iw = xy.x['workload']

            Xp = self.objectives['mf'].data['platform'][ip]
            Xw = self.objectives['mf'].data['workload'][iw]
            X = jnp.concatenate([Xp, Xw], axis=1)

            y_hat = self.nn.apply(params["mf"], X)
            for k in xy.x:
                if k.startswith("interference"):
                    Xw_if = self.objectives['mf'].data['workload'][xy.x[k]]
                    X_if = jnp.concatenate([Xp, Xw, Xw_if], axis=1)
                    y_hat += self.nn.apply(params["if"], X_if)

            return types.Predictions(y_true=xy.y, y_hat=y_hat)

        return {k: _evaluate(v) for k, v in data.items()}

    @classmethod
    def from_config(
        cls, objectives: ObjectiveSet, layers: list[int] = [256, 256],
        loss_class: str = "Squared", loss_args: dict = {},
        optimizer: str = "adamax", optimizer_args: dict = {}, **kwargs
    ) -> "Monolith":
        """Create from configuration."""
        loss_func = getattr(loss, loss_class).from_config(**loss_args)
        opt = getattr(optax, optimizer)(**optimizer_args)
        return cls(objectives, loss_func, optimizer=opt, layers=layers)
