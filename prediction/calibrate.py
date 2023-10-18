"""Split Conformal Regression Implementation."""

from functools import partial
from jax import vmap
from jax import numpy as jnp

from jaxtyping import Array, Float

from .loss import PercentError


def coverage(
    y_true: Float[Array, "N"], y_hat: Float[Array, "N options"]
) -> Float[Array, "options"]:
    """Determine actual miscoverage ratio of a predicted bound."""
    return jnp.mean(y_true[:, None] < y_hat, axis=0)


def margin(
    y_true: Float[Array, "N"], y_hat: Float[Array, "N options"],
    alpha: Float[Array, ""]
) -> Float[Array, "options"]:
    """Calculate margin required to conformalize coverage."""
    quantile = (1 - alpha) * (1 + 1 / y_true.shape[0])
    return jnp.quantile(y_true[:, None] - y_hat, quantile, axis=0)


def width(
    y_true: Float[Array, "N"], y_hat: Float[Array, "N options"],
    log: bool = True
) -> Float[Array, "options"]:
    """Calculate mean overprovisioning interval width.

    Since we are taking a 1-sided interval in the log domain, widths are
    technically always infinite. We instead define the following metric::

        E[max(0, (y_hat - y_true) / y_true]

    Which is a measure of the overprovisioning required by the method to
    reach the target miscoverage ratio.
    """
    return jnp.mean(
        jnp.maximum(0, PercentError(log=log)(y_true, y_hat)), axis=0)


def calibrate(
    y_true: Float[Array, "N"], y_hat: Float[Array, "N options"],
    alpha: Float[Array, "Q"], log: bool = True
) -> tuple[Float[Array, "options"], Float[Array, "options"]]:
    """Calculate calibration margins and interval widths for each output.

    Parameters
    ----------
    y_true, y_hat: Actual/predicted values.
    alpha: Calibration targets.
    log: Whether y_true/y_hat are in the log domain.

    Returns
    -------
    margin: Calibration margins to reach the target miscoverage alpha(s).
    width: Mean overprovisioning interval width for each calibrated output.
    """
    if len(y_hat.shape) == 1:
        y_hat = y_hat[:, None]

    _margin = vmap(partial(
        margin, y_true=y_true, y_hat=y_hat))(alpha=alpha)
    _width = vmap(partial(
        width, y_true=y_true, log=log
    ))(y_hat=y_hat[None, :, :] + _margin[:, None, :])

    return _margin, _width
