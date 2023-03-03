"""Train/Val/Test Splitting."""

from jax import numpy as jnp
from jax import random, vmap

from jaxtyping import UInt32, Array, Shaped


def keys(key: UInt32[Array, "2"], n: int) -> UInt32[Array, "k 2"]:
    """Split PRNGKey into jnp.array of keys."""
    return jnp.array(random.split(key, n))


def split(
    key: UInt32[Array, "2"], data: Shaped[Array, "n ..."], split: int = 100
) -> tuple[Shaped[Array, "n1 ..."], Shaped[Array, "n2 ..."]]:
    """Split off an exact number of data points, where each point is a row.

    Parameters
    ----------
    key: PRNGKey.
    data: data to split.
    split: exact number of data points to split off.

    Returns
    -------
    train: main dataset.
    test: data points that were split off.
    """
    idx = random.permutation(key, jnp.arange(data.shape[0]))
    return data[idx[:-split]], data[idx[-split:]]


def crossval(
    key: UInt32[Array, "2"], data: Shaped[Array, "n ..."], k: int = 10
) -> tuple[Shaped[Array, "k n1 ..."], Shaped[Array, "k n2 ..."]]:
    """Generate k-fold cross-validation splits.

    The split forms a new axis in position 0 (result[i] denotes split i).

    Parameters
    ----------
    key: PRNGKey.
    data: input data to split.
    k: number of folds.

    Returns
    -------
    train: training split with size (k-1)/k
    test: test split with size 1/k
    """
    size = int(data.shape[0] / k)
    idx = random.permutation(key, jnp.arange(data.shape[0]))
    orders = data[idx][(
        jnp.arange(data.shape[0]).reshape(1, -1)
        + jnp.arange(k).reshape(-1, 1) * size
    ) % data.shape[0]]

    train = orders[:, size:]
    test = orders[:, :size]
    return train, test


def batch(
    key: UInt32[Array, "2"], data: Shaped[Array, "n ..."], batch: int = 64
) -> Shaped[Array, "b ..."]:
    """Sample IID batch along axis 0."""
    if data is None:
        return None
    indices = random.randint(
        key, shape=(batch,), minval=0, maxval=data.shape[0])
    return data[indices]
