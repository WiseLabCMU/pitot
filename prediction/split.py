"""Train/Val/Test Splitting."""

from jax import numpy as jnp
from jax import random
from jax import tree_util

from jaxtyping import UInt32, Array, Shaped, PyTree, Integer
from beartype.typing import Optional, Union


def keys(key: UInt32[Array, "2"], n: int) -> UInt32[Array, "k 2"]:
    """Split PRNGKey into jnp.array of keys."""
    return jnp.array(random.split(key, n))


def tree_size(data: PyTree[Shaped[Array, "n ..."]]) -> int:
    """Get size of PyTree."""
    sizes = [x.shape[0] for x in tree_util.tree_leaves(data)]
    assert all(x == sizes[0] for x in sizes)
    return sizes[0]


def tree_index(
    data: PyTree[Shaped[Array, "n ..."]],
    indices: Union[slice, Integer[Array, "..."]]
) -> PyTree[Shaped[Array, "..."]]:
    """Index into PyTree."""
    return tree_util.tree_map(lambda x: x[indices], data)


def split(
    key: UInt32[Array, "2"], data: PyTree[Shaped[Array, "n ..."]],
    split: int = 100
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
    idx = random.permutation(key, jnp.arange(tree_size(data)))
    train = tree_index(data, idx[:-split])
    val = tree_index(data, idx[-split:])
    return (train, val)


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
    size = tree_size(data)
    cv_size = int(size / k)

    shuffled: Integer[Array, "n"] = random.permutation(key, jnp.arange(size))
    splits: Integer[Array, "k n"] = shuffled[(
        jnp.arange(size).reshape(1, -1)
        + jnp.arange(k).reshape(-1, 1) * cv_size
    ) % size]

    train = tree_index(data, splits[:, cv_size:])
    val = tree_index(data, splits[:, :cv_size])
    return (train, val)


def batch(
    key: UInt32[Array, "2"], data: Shaped[Array, "n ..."],
    batch: Optional[int] = 64
) -> Shaped[Array, "b ..."]:
    """Sample IID batch along axis 0.

    If batch is None, returns full batch.

    Parameters
    ----------
    key: JAX PRNGKey state.
    data: Data points, where each row (axis 0) is a data point.
    batch: Target batch size.

    Returns
    -------
    Sampled data points with the same shape as `data.shape[1:]`.
    """
    if batch is None:
        return data
    else:
        indices = random.randint(
            key, shape=(batch,), minval=0, maxval=data.shape[0])
        return data[indices]
