"""Dataset for prediction."""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp

from jax import random, vmap
from sklearn.decomposition import PCA


IndexSplit = namedtuple("IndexSplit", ['train', 'val', 'test', 'k', 'n', 'p'])


class Dataset:
    """Dataset.

    Parameters
    ----------
    data : str or dict
        Source data, or filepath to npz file containing data.
    key : callable
        Callable that fetches the target value from the archive.
    device : torch.device
        Device to place data on.
    val : float
        Proportion of dataset to reserve for validation split.
    test : float
        Proportion of dataset to reserve for test split.
    """

    def __init__(
            self, data="polybench.npz", key=lambda x: x['mean'], device=None):

        if isinstance(data, str):
            data = dict(np.load(data))
        self.data = data
        self._data_key = key(data)
        self.matrix = jnp.log(self._data_key)
        self.rms = jnp.sqrt(jnp.mean(jnp.square(self.matrix)))
        if device is not None:
            self.matrix_torch = self.matrix_torch.to(device)

        opcodes_nonzero = np.sum(data['opcodes'], axis=0) > 0
        _opcodes = data['opcodes'][:, opcodes_nonzero]
        self.opcodes = jnp.log(_opcodes.astype(jnp.float32) + 1)
        self.opcodes_pca = jnp.array(PCA().fit_transform(self.opcodes))
        self.opcode_names = np.where(opcodes_nonzero)

        self.files = data['files']
        self.runtimes = data['runtimes']
        self.size = jnp.sum(self._data_key > 0)

    def grid(self):
        """Create (x, y) grid pairs."""
        x, y = jnp.meshgrid(
            jnp.arange(self.matrix.shape[0]), jnp.arange(self.matrix.shape[1]))
        return jnp.stack([x.reshape(-1), y.reshape(-1)]).T

    def split_iid(self, key, p=0.5, n=1):
        """Create Data split between train and test sets."""
        def _inner(_key):
            xy = self.grid()
            idx = random.permutation(_key, jnp.arange(xy.shape[0]))

            train_size = int(xy.shape[0] * p)
            return xy[idx[:train_size]], xy[idx[train_size:]]

        _, *keys = random.split(key, n + 1)
        train, test = vmap(_inner)(jnp.array(keys))
        return train, test

    def split_kfold(self, key, k=2, n=1):
        """Perform k-fold minor data split into train and test sets.

        Assignments are exact, so each split will have exactly the same split
        sizes. Remainder points will stay in the test set and not be included
        in any of the train sets.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root PRNGKey for this operation.
        k : int
            Number of splits for test set.
        n : int
            Number of repetitions.
        """
        def _inner(_key):
            xy = self.grid()
            idx = random.permutation(_key, jnp.arange(xy.shape[0]))
            size = int(xy.shape[0] / k)

            orders = xy[idx[(
                jnp.arange(xy.shape[0]).reshape(1, -1)
                + jnp.arange(k).reshape(-1, 1) * size
            ) % xy.shape[0]]]
            return (orders[:, :size, :], orders[:, size:, :])

        _, *keys = random.split(key, n + 1)
        train, test = vmap(_inner)(jnp.array(keys))
        return (
            train.reshape(-1, *train.shape[2:]),
            test.reshape(-1, *test.shape[2:]))

    @staticmethod
    def split_crossval(data, split=10):
        """Split training set into train and validation sets for k-fold CV.

        Assumes the train set is already shuffled.
        """
        def _inner(row):
            size = int(row.shape[0] / split)
            orders = row[(
                jnp.arange(row.shape[0]).reshape(1, -1)
                + jnp.arange(split).reshape(-1, 1) * size
            ) % row.shape[0]]
            return (orders[:, size:, :], orders[:, :size, :])

        train, val = vmap(_inner)(data)
        return train, val

    def split(self, key, splits=100, p=0.25, kval=24):
        """Generate data splits using n sets of k-fold splits or IID splits.

        Returns index arrays with the following shape:

            (splits, kval, samples, 2)

        Where:
          - splits is the number of replicates (with train+val/test splits)
          - kval is the k-fold cross validation performed on each replicate
          - samples is the number of samples in that split
          - last axis specifies (x, y) of the sample

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root PRNGKey for this operation.
        splits : int
            Number of splits to generate.
        p : int
            Split proportion. If approximately 1/k or 1 - 1/k, uses k-fold;
            otherwise, uses IID splits.
        kval : int
            Number of splits for train set for k-fold CV.

        Returns
        -------
        IndexSplit
            train, val, and test splits, along with metadata.
        """
        if p < 0.5 and np.abs(int(1 / p) - 1 / p) < 0.05:
            k = round(1 / p)
            n = int(splits / k)
            train, test = self.split_kfold(key, k, n)
        elif p > 0.5 and np.abs(int(1 / (1 - p)) - 1 / (1 - p)) < 0.05:
            k = round(1 / (1 - p))
            n = int(splits / k)
            test, train = self.split_kfold(key, k, n)
        else:
            k = None
            n = splits
            train, test = self.split_iid(key, p=p, n=splits)

        train, val = self.split_crossval(train, split=kval)
        test = jnp.tile(
            test.reshape([test.shape[0], 1, *test.shape[1:]]), [1, kval, 1, 1])
        return IndexSplit(train=train, val=val, test=test, p=p, k=k, n=n)

    def to_mask(self, xy):
        """Convert indices to mask."""
        return jnp.zeros_like(self.matrix).at[xy[:, 0], xy[:, 1]].set(1.)

    def index(self, indices):
        """Index into data."""
        return self.matrix[indices[:, 0], indices[:, 1]]

    def loss(self, pred, indices):
        """Compute l2-log loss."""
        y_true = self.index(indices)
        return jnp.mean(jnp.square(pred - y_true))
