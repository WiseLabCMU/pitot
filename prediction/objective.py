"""Prediction Objectives."""

from jax import numpy as jnp

from beartype.typing import NamedTuple, Optional, Union
from jaxtyping import Float32, Array, Integer

from .dataset import Dataset


#: Matrix Factorization Predictions. Can be a matrix, array, k-fold replicates
#: of matrix, or k-fold replicates of array.
MFPrediction = Union[
    Float32[Array, "n"],
    Float32[Array, "nx ny"],
    Float32[Array, "k nx ny"]
]


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
    normalize: normalize loss by magnitude.
    """

    x: Integer[Array, "N k"]
    y: Float32[Array, "N"]
    batch_size: Optional[int]
    weight: Optional[float]
    log: bool
    name: str
    save: str
    normalize: bool = False

    @classmethod
    def from_config(cls, dataset: Dataset, config: dict) -> "Objective":
        """Create objective from configuration.

        "xkey" and "ykey" attributes denote where to fetch `x` and `y` from.
        """
        x = getattr(dataset, config["xkey"])
        y = getattr(dataset, config["ykey"])
        kwargs = {k: v for k, v in config.items() if k not in {"xkey", "ykey"}}
        return cls(x=x, y=y, **kwargs)

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

    def index(  # type: ignore
        self, pred: MFPrediction, idx: Integer[Array, "n"]
    ) -> tuple[Float32[Array, "n"], Float32[Array, "n"]]:
        """Index into dataset y values (and predictions).

        NOTE: There's an edge case here if `len(idx) == matrix.shape[1]`.
        """
        x = self.x[idx]
        y = self.y[idx]

        # Index form
        if pred.shape[-1] == idx.shape[0]:
            if len(pred.shape) == 2:
                pred = jnp.mean(pred, axis=0)
            return pred, y
        # Matrix form
        else:
            if len(pred.shape) == 3:
                pred = jnp.mean(pred, axis=0)
            return pred[x[:, 0], x[:, 1]], y

    def loss(self, pred: Float32[Array, "..."], idx: Integer[Array, "n"]):
        """Compute (weighted) loss from data indices."""
        if self.weight is None:
            return 0.0
        elif self.normalize:
            return jnp.mean(
                jnp.square((pred - self.y[idx]) / self.y[idx])) * self.weight    
        else:
            return jnp.mean(jnp.square(pred - self.y[idx])) * self.weight

    def error(self, pred: MFPrediction, idx: Integer[Array, "n"]):
        """Get full prediction error."""
        pred, actual = self.index(pred, idx)
        return pred - actual

    def perror(self, pred: MFPrediction, idx: Integer[Array, "n"]):
        """Percent error."""
        if self.log:
            return jnp.exp(self.error(pred, idx)) - 1
        else:
            pred, actual = self.index(pred, idx)
            return (pred - actual) / actual

    def rmse(self, pred: MFPrediction, idx: Integer[Array, "n"]):
        """RMSE log error."""
        return jnp.sqrt(jnp.mean(jnp.square(self.error(pred, idx))))

    def mae(self, pred: MFPrediction, idx: Integer[Array, "n"]):
        """Mean absolute error."""
        return jnp.mean(jnp.abs(self.error(pred, idx)))

    def mape(self, pred: MFPrediction, idx: Integer[Array, "n"]):
        """Mean absolute percent error."""
        return jnp.mean(jnp.abs(self.perror(pred, idx)))
