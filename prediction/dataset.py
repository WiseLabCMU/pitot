"""Dataset for prediction."""

import numpy as np
from jax import numpy as jnp

from beartype.typing import NamedTuple
from jaxtyping import Float32, Array, Integer, Bool


class Dataset(NamedTuple):
    """Matrix Factorization Dataset.

    NOTE: we assume that there are fewer than 256 modules / platforms.

    Attributes
    ----------
    data: execution time data matrix.
    x: valid (platform, module) indices in the matrix.
    y: valid execution times.
    if_x: interference (platform, module, interferer...) indices.
    if_y: interference execution time, if available.
    x_p: platform side information.
    x_m: module side information (log-opcodes).
    log: whether this dataset is in log-space.
    """

    data: Float32[Array, "Np Nm"]
    x: Float32[Array, "Nf 2"]
    y: Float32[Array, "Nf"]
    if_x: Float32[Array, "Ni k"]
    if_y: Float32[Array, "Ni"]
    x_p: Float32[Array, "Np Dp"]
    x_m: Float32[Array, "Nm Dm"]
    log: bool

    def to_mask(self, ij: Integer[Array, "n d"]) -> Bool[Array, "Np Nm"]:
        """Convert list of indices to mask."""
        return jnp.zeros_like(
            self.data, dtype=jnp.bool_).at[ij[:, 0], ij[:, 1]].set(True)

    @classmethod
    def from_npz(
        cls, path: str = "data.npz", log: bool = True, offset: float = 25000.,
    ) -> "Dataset":
        """Load dataset from file.

        Parameters
        ----------
        path: file path. Should have 'data', 'platform_data', 'module_data'
            keys.
        log: whether to apply log transform to the data.
        offset: multiplicative data offset to divide by.
        """
        def transform(x):
            if log:
                return jnp.log(x) - jnp.log(offset)
            else:
                x = x / offset
                x[x == 0] = np.nan
                return x

        npz = np.load(path)
        with np.errstate(divide='ignore'):
            data = jnp.array(transform(npz["data"]), dtype=jnp.float32)
            x = jnp.stack(jnp.where(jnp.isfinite(data))).T
            y = data[x[:, 0], x[:, 1]]
            x_p = jnp.array(npz["platform_data"], dtype=jnp.float32)
            x_m = jnp.array(jnp.log(npz["module_data"] + 1), dtype=jnp.float32)

        if "if_data" in npz:
            if_y = transform(npz["if_data"])[:y.size]
            if_x = jnp.concatenate([
                npz["if_platform"].reshape(-1, 1),
                npz["if_module"].reshape(-1, 1),
                npz["if_interferer"]], axis=1)[:y.size]
        else:
            if_y, if_x = None, None

        return cls(
            data=jnp.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
            x=x, y=y, if_x=if_x, if_y=if_y, x_p=x_p, x_m=x_m, log=log)
