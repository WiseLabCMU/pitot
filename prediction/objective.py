"""Prediction Objectives."""

from jax import numpy as jnp

from beartype.typing import NamedTuple, Optional, Union
from jaxtyping import Float32, Array, Integer


class Objective(NamedTuple):
    """Training objective.

    Attributes
    ----------
    x: input indices.
    y: measured execution time.
    batch_size: batch size when sampling for SGD; full batch if None.
    weight: weight of this objective.
    log: whether this objective uses log data.
    name: name of this objective.
    save: if not None, save the test predictions to this key.
    """

    x: Integer[Array, "N k"]
    y: Float32[Array, "N"]
    batch_size: Optional[int]
    weight: Optional[float]
    log: bool
    name: str
    save: str

    @property
    def size(self) -> int:
        """Dataset size."""
        return self.y.shape[0]

    @property
    def indices(self) -> Integer[Array, "N"]:
        """Possible data indices `y.shape[0]`."""
        u16 = self.y.shape[0] < jnp.iinfo(jnp.uint16).max
        return jnp.arange(
            self.y.shape[0], dtype=jnp.uint16 if u16 else jnp.uint32)

    def loss(self, pred: Float32[Array, "n"], idx: Integer[Array, "n"]):
        """Compute (weighted) loss from data indices."""
        if self.weight is None:
            return 0.0
        else:
            return jnp.mean(jnp.square(pred - self.y[idx])) * self.weight

    def rmse(self, pred: Float32[Array, "n"], idx: Integer[Array, "n"]):
        """RMSE log error; alias for sqrt(loss)."""
        return jnp.sqrt(self.loss(pred=pred, idx=idx))

    def mae(self, pred: Float32[Array, "n"], idx: Integer[Array, "n"]):
        """Mean absolute error."""
        return jnp.mean(jnp.abs(pred - self.y[idx]))

    def mape(self, pred: Float32[Array, "n"], idx: Integer[Array, "n"]):
        """Mean absolute percent error."""
        if self.log:
            return jnp.mean(jnp.abs(jnp.exp(pred - self.y[idx]) - 1))
        else:
            return jnp.mean(jnp.abs(pred - self.y[idx]) / self.y[idx])

    def error(self, pred: Float32[Array, "n"], idx: Integer[Array, "n"]):
        """Get full prediction error."""
        return pred - self.x[idx]
