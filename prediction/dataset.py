"""Dataset for prediction."""

import numpy as np
from jax import numpy as jnp

from beartype.typing import NamedTuple, Optional, Callable, Union
from jaxtyping import Float32, Array, Bool, Integer, Shaped


DataTransform = Callable[[Shaped[Array, "nx ny"]], Shaped[Array, "nx ny"]]


class Dataset(NamedTuple):
    """Matrix Factorization Dataset.

    Attributes
    ----------
    data: (platforms x modules) data matrix.
    mask: valid samples in the matrix.
    x_p: platform side information.
    x_m: module side information (log-opcodes).
    """

    data: Float32[Array, "Np Nm"]
    mask: Bool[Array, "Np Nm"]
    x_p: Float32[Array, "Np Dp"]
    x_m: Float32[Array, "Nm Dm"]
    log: bool

    @classmethod
    def from_npz(
        cls, path: str = "data.npz", log: bool = True, offset: float = 25000.
    ) -> "Dataset":
        """Load dataset from file.

        Parameters
        ----------
        path: file path. Should have 'data', 'platform_data',
        'module_data' keys.
        log: whether to apply log to the data.
        offset: multiplicative data offset to divide by.
        """
        def transform(x):
            if log:
                return np.log(x) - np.log(offset)
            else:
                x = x / offset
                x[x == 0] = np.nan
                return x

        npz = np.load(path)
        with np.errstate(divide='ignore'):
            data = jnp.array(transform(npz["data"]), dtype=jnp.float32)
            x_p = jnp.array(npz["platform_data"], dtype=jnp.float32)
            x_m = jnp.array(np.log(npz["module_data"] + 1), dtype=jnp.float32)
        return cls(
            data=jnp.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
            mask=jnp.isfinite(data), x_p=x_p, x_m=x_m, log=log)

    @property
    def indices(self) -> Integer[Array, "n 2"]:
        """Get (i, j) valid platform-module pairs."""
        return jnp.stack(jnp.where(self.mask)).T

    def index(
        self, ij: Integer[Array, "n 2"],
        pred: Optional[
            Union[Float32[Array, "Np Nm"], Float32[Array, "n"]]] = None
    ) -> Union[
        Float32[Array, "n"], tuple[Float32[Array, "n"], Float32[Array, "n"]]
    ]:
        """Index into matrix (and optionally index predictions)."""
        if pred is None:
            return self.data[ij[:, 0], ij[:, 1]]
        elif pred.shape == self.data.shape:
            return self.index(ij), pred[ij[:, 0], ij[:, 1]]
        else:
            return self.index(ij), pred

    def to_mask(self, ij: Integer[Array, "n 2"]) -> Bool[Array, "Np Nm"]:
        """Convert list of indices to mask."""
        return jnp.zeros_like(
            self.data, dtype=jnp.bool_).at[ij[:, 0], ij[:, 1]].set(True)

    def loss(
        self, pred: Union[Float32[Array, "Np Nm"], Float32[Array, "n"]],
        indices: Integer[Array, "n 2"],
    ) -> Float32[Array, ""]:
        """Compute loss."""
        actual, pred = self.index(indices, pred)
        return jnp.mean(jnp.square(pred - actual))

    def rmse(
        self, pred: Union[Float32[Array, "Np Nm"], Float32[Array, "n"]],
        indices: Integer[Array, "n 2"],
    ) -> Float32[Array, ""]:
        """RMSE log error; alias for sqrt(loss)."""
        return jnp.sqrt(self.loss(pred=pred, indices=indices))

    def mae(
        self, pred: Union[Float32[Array, "Np Nm"], Float32[Array, "n"]],
        indices: Integer[Array, "n 2"],
    ) -> Float32[Array, ""]:
        """Mean absolute error."""
        pred, actual = self.index(indices, pred)
        return jnp.mean(jnp.abs(pred - actual))

    def mape(
        self, pred: Union[Float32[Array, "Np Nm"], Float32[Array, "n"]],
        indices: Integer[Array, "n 2"],
    ) -> Float32[Array, ""]:
        """Mean absolute percent error."""
        pred, actual = self.index(indices, pred)
        if self.log:
            return jnp.mean(jnp.abs(jnp.exp(pred - actual) - 1))
        else:
            return jnp.mean(jnp.abs(pred - actual) / actual)

    def error(
        self, pred: Union[Float32[Array, "Np Nm"], Float32[Array, "n"]],
        indices: Integer[Array, "n 2"],
    ) -> Float32[Array, ""]:
        """Get full prediction error."""
        pred, actual = self.index(indices, pred)
        return pred - actual
