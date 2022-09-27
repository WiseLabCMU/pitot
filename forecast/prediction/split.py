"""Train/Val/Test Splitting."""

from jax import numpy as jnp
from jax import random


def keys(key, n):
    """Split PRNGKey into jnp.array of keys."""
    return jnp.array(random.split(key, n))


def diagonal(dims, offset=0):
    """Matrix Completion: get elements from the (wrapped) diagonal.

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


def iid(key, data=None, train=0):
    """Generate IID split."""
    idx = random.permutation(key, jnp.arange(data.shape[0]))
    if data is None:
        return idx[:train], idx[train:]
    else:
        return data[idx[:train]], data[idx[train:]]


def shuffle(key, data):
    """Shuffle data along first axis."""
    idx = random.permutation(key, jnp.arange(data.shape[0]))
    return data[idx]


def at_least_one(key, offset=0, dim=None, train=100):
    """At least one element of each row is included in the train set."""
    train_0, test_0 = diagonal(dim, offset)
    train_1, test_1 = iid(key, data=test_0, train=train)
    return jnp.concatenate([train_0, train_1]), test_1


def crossval(key, data, split=10):
    """Split training set into train and validation sets for k-fold CV.

    Creates an additional axis with dimension split: [...] -> [split, ...].
    """
    size = int(data.shape[0] / split)
    orders = shuffle(key, data)[(
        jnp.arange(data.shape[0]).reshape(1, -1)
        + jnp.arange(split).reshape(-1, 1) * size
    ) % data.shape[0]]
    return (orders[:, size:, :], orders[:, :size, :])
