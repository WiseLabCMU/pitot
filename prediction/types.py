"""Data types for runtime matrix completion algorithms."""

from beartype.typing import NamedTuple
from jaxtyping import Array, Float, PyTree, UInt


class Data(NamedTuple):
    """Data batch with indices, runtime."""

    x: dict[str, UInt[Array, "batch"]]
    y: Float[Array, "batch"]


class Predictions(NamedTuple):
    """Predictions and observed values."""

    y_true: Float[Array, "batch"]
    y_hat: Float[Array, "batch objs"]


class TrainState(NamedTuple):
    """Training state."""

    params: PyTree
    opt_state: PyTree
