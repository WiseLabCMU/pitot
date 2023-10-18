"""Prediction objectives/metrics."""

from jax import numpy as jnp
from jaxtyping import Array, Float32
from beartype.typing import Optional, Union, Sequence


class Loss:
    """Loss base class."""

    N: int = 1
    weight: Float32[Array, "1"] = jnp.ones(1)

    def __call__(
        self, y_true: Float32[Array, "batch"], y_hat: Float32[Array, "batch N"]
    ) -> Float32[Array, "batch N"]:
        """Compute loss value."""
        raise NotImplementedError()

    @classmethod
    def from_config(cls) -> "Loss":
        """Create from configuration."""
        return cls()


class PercentError(Loss):
    """Percent error loss.

    Parameters
    ----------
    log: whether predictions are in the log domain.
    """

    weight: Float32[Array, "1"] = jnp.ones(1)

    def __init__(self, log: bool = True) -> None:
        self.log = log

    def __call__(
        self, y_true: Float32[Array, "batch"], y_hat: Float32[Array, "batch 1"]
    ) -> Float32[Array, "batch 1"]:
        """Compute absolute percent error."""
        if self.log:
            return jnp.abs(jnp.exp(y_true[:, None] - y_hat) - 1)
        else:
            return jnp.abs((y_true[:, None] - y_hat) / y_true[:, None])

    def __repr__(self) -> str:
        """Get descriptive name."""
        return "PercentError(log={})".format(self.log)


class Squared(Loss):
    """Generic l2 loss."""

    weight: Float32[Array, "1"] = jnp.ones(1)

    @staticmethod
    def __call__(
        y_true: Float32[Array, "batch"], y_hat: Float32[Array, "batch 1"]
    ) -> Float32[Array, "batch 1"]:
        """Compute squared l2 loss."""
        return jnp.square(y_true[:, None] - y_hat)

    def __repr__(self) -> str:
        """Get descriptive name."""
        return "Squared()"


class Pinball(Loss):
    """Pinball loss for quantile regression.

    NOTE: in quantile regression, alpha usually refers to the quantile
    (lower alpha = smaller prediction), but in conformal prediction, alpha
    refers to the miscoverage ratio, which in our case is right-sided
    (lower alpha = bigger prediction). We use our conformal regression
    convention here to reduce confusion.

    Parameters
    ----------
    alpha: target quantile (larger quantile = lower prediction)
    """

    def __init__(
        self, alpha: Float32[Array, "n"],
        weight: Optional[Float32[Array, "n"]] = None
    ) -> None:
        if isinstance(alpha, float):
            alpha = jnp.array([alpha], dtype=jnp.float32)
        if weight is None:
            weight = jnp.ones(alpha.shape[0]) / alpha.shape[0]

        self.alpha = alpha
        self.weight = weight
        self.N = alpha.shape[0]

    def __call__(
        self, y_true: Float32[Array, "batch"],
        y_hat: Float32[Array, "batch N"],
    ) -> Float32[Array, "batch N"]:
        """Compute pinball loss(es)."""
        err = y_hat - y_true[:, None]
        return (
            (1 - self.alpha[None, :]) * (-err) * (err < 0)
            +   (self.alpha[None, :]) *   err  * (err >= 0))

    @classmethod
    def from_config(
        cls, quantiles: Sequence[Union[int, float]] = [90, 95, 99]
    ) -> "Loss":
        """Create from configuration."""
        return cls(
            alpha=1 - jnp.array(quantiles) / 100,
            weight=jnp.ones(len(quantiles)) / len(quantiles))

    def __repr__(self) -> str:
        """Get descriptive name."""
        return "Pinball({})".format(
            ', '.join(["{:.2f}".format(p) for p in self.alpha]))
