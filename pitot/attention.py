"""Attention-based interference model."""

import haiku as hk
import optax
import jax
from jax import numpy as jnp

from jaxtyping import PyTree, PRNGKeyArray
from beartype.typing import Optional

from prediction import MatrixCompletionModel, loss, ObjectiveSet, types, Split
from prediction.utils import array_unpack


def _mlp(layers: list[int], dim: int) -> hk.Module:
    _layers = sum([
        [hk.Linear(d, name="mlp_{}".format(i)), jax.nn.gelu]  # type: ignore
        for i, d in enumerate(layers)
    ], []) + [hk.Linear(dim, name="mlp_out")]                 # type: ignore
    return hk.Sequential(_layers, name="mlp")                 # type: ignore


class Attention(MatrixCompletionModel):
    """Attention-based interference model.

    Parameters
    ----------
    layers: neural network layers.
    """

    def __init__(
        self, objectives: ObjectiveSet, losses: loss.Loss,
        optimizer: Optional[optax.GradientTransformation] = None,
        embedding_layers: list[int] = [256, 256], attention_dim: int = 64,
        value_dim: int = 64, output_layers: list[int] = [64]
    ) -> None:
        self.attention_dim = attention_dim
        self.value_dim = value_dim

        def _module(layers, dim):
            def forward(*args, **kwargs):
                return _mlp(layers=layers, dim=dim)(*args, **kwargs)

            return hk.without_apply_rng(hk.transform(forward))

        self.nn = _module(embedding_layers, attention_dim + losses.N)
        self.embedding = _module(embedding_layers, attention_dim + value_dim)
        self.output = _module(output_layers, losses.N)

        super().__init__(
            objectives=objectives, losses=losses, optimizer=optimizer,
            name="Attention")

    def _init(
        self, key: PRNGKeyArray, splits: dict[str, Split]
    ) -> PyTree:
        """Get model parameters."""
        dim = (
            self.objectives['mf'].data['platform'].shape[1]
            + self.objectives['mf'].data['workload'].shape[1])
        return {
            "mf": self.nn.init(key, jnp.zeros((2, dim))),
            "embedding": self.embedding.init(key, jnp.zeros((2, dim))),
            "output": self.output.init(key, jnp.zeros((2, self.value_dim)))}

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
            iw = xy.x['workload']
            ip = xy.x['platform']

            Xw = self.objectives['mf'].data['workload'][iw]
            Xp = self.objectives['mf'].data['platform'][ip]

            query, y_hat = array_unpack(
                self.nn.apply(params["mf"], jnp.concatenate([Xp, Xw], axis=1)),
                (self.attention_dim,), (self.loss_func.N,))

            KV = []
            for k in xy.x:
                if k.startswith("interference"):
                    Xw_if = self.objectives['mf'].data['workload'][xy.x[k]]
                    X_if = jnp.concatenate([Xp, Xw_if], axis=1)
                    KV.append(array_unpack(
                        self.embedding.apply(params["embedding"], X_if),
                        (self.attention_dim,), (self.value_dim,)))

            if len(KV) > 0:
                attention = jnp.zeros(())
                for key, value in KV:
                    attention += value * jnp.sum(key * query, axis=1)[:, None]
                y_hat += self.output.apply(params["output"], attention)

            return types.Predictions(y_true=xy.y, y_hat=y_hat)

        return {k: _evaluate(v) for k, v in data.items()}

    @classmethod
    def from_config(
        cls, objectives: ObjectiveSet,
        embedding_layers: list[int] = [256, 256], attention_dim: int = 64,
        value_dim: int = 64, output_layers: list[int] = [64],
        loss_class: str = "Squared", loss_args: dict = {},
        optimizer: str = "adamax", optimizer_args: dict = {}, **kwargs
    ) -> "Attention":
        """Create from configuration."""
        loss_func = getattr(loss, loss_class).from_config(**loss_args)
        opt = getattr(optax, optimizer)(**optimizer_args)
        return cls(
            objectives, loss_func, optimizer=opt,
            embedding_layers=embedding_layers, attention_dim=attention_dim,
            value_dim=value_dim, output_layers=output_layers)
