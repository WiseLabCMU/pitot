"""Train/Val/Test Splitting."""

from functools import partial
from jax import numpy as jnp
from jax import random, vmap


def keys(key, n):
    """Split PRNGKey into jnp.array of keys."""
    return jnp.array(random.split(key, n))


def diagonal(dims, offset=0):
    """2D: get elements from the (wrapped) diagonal.

    Assumes cols <= rows.
    """
    rows, cols = dims
    assert(cols <= rows)

    train = jnp.array((
        # Each row ...
        jnp.arange(rows),
        # Diagonal starting from top left with offset
        (jnp.arange(rows) + offset) % cols
    )).T

    test = jnp.array((
        # Each row ...
        jnp.repeat(jnp.arange(rows), cols - 1), (
            # All values other than the diagonal:
            (
                # Start on the diagonal
                jnp.arange(rows).reshape(-1, 1)
                # Move by (1...col)
                + 1 + jnp.arange(cols - 1).reshape(1, -1)
                # Add offset to shift diagonal right
                + offset
            )
            # Modulo to bring it back
            % cols
        ).reshape(-1)
    )).T

    return train, test


def at_least_one(key, offset=0, dim=None, train=100):
    """2D: at least one entry in each row is included in the train set."""
    train_0, test_0 = diagonal(dim, offset)

    idx = random.permutation(key, jnp.arange(test_0.shape[0]))
    train_1 = test_0[idx[:train - dim[0]]]
    test_1 = test_0[idx[train - dim[0]:]]

    return jnp.concatenate([train_0, train_1]), test_1


def vmap_at_least_one(key, dim=None, train=100, replicates=100):
    """vmap-2D: create at_least_one splits."""
    return vmap(partial(at_least_one, dim=dim, train=train))(
        keys(key, replicates), jnp.arange(replicates) % dim[0])


def crossval(key, data, split=10):
    """ND: split training set into train and validation sets for k-fold CV.

    Creates an additional axis with dimension split: [...] -> [split, ...].
    """
    size = int(data.shape[0] / split)
    idx = random.permutation(key, jnp.arange(data.shape[0]))
    orders = data[idx][(
        jnp.arange(data.shape[0]).reshape(1, -1)
        + jnp.arange(split).reshape(-1, 1) * size
    ) % data.shape[0]]
    return (orders[:, size:], orders[:, :size])


def vmap_crossval(key, data, split=10):
    """vmap-ND: split training sets into train and validation sets."""
    return vmap(partial(crossval, split=split))(keys(key, data.shape[0]), data)


def batch(key, data, batch=64):
    """1D: sample batch along axis 0."""
    if data is None:
        return None
    indices = random.randint(
        key, shape=(batch,), minval=0, maxval=data.shape[0])
    return data[indices]


def iid(key, dim=None, train=100):
    """1D: generate IID data split."""
    idx = random.permutation(key, jnp.arange(dim))
    return idx[:train], idx[train:]


def vmap_iid(key, dim=None, train=100, replicates=100):
    """vmap-ND: generate IID data splits."""
    return vmap(partial(iid, dim=dim, train=train))(keys(key, replicates))
