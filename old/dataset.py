"""Dataset for prediction."""

from functools import partial
from collections import namedtuple

import numpy as np
from jax import numpy as jnp

from jax import random, vmap
from sklearn.decomposition import PCA


IndexSplit = namedtuple("IndexSplit", ['train', 'val', 'test'])


class Dataset:
    """Dataset.

    Parameters
    ----------
    data : str or dict
        Source data, or filepath to npz file containing data.
    device : torch.device
        Device to place data on.
    val : float
        Proportion of dataset to reserve for validation split.
    test : float
        Proportion of dataset to reserve for test split.
    """

    def __init__(
            self, data="data/polybench/20-40.npz", device=None):

        if isinstance(data, str):
            data = dict(np.load(data))
        self.data = data
        self.matrix = jnp.log(data['runtime'])
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
        self.size = jnp.sum(data['runtime'] > 0)

    def grid(self):
        """Create (x, y) grid pairs."""
        x, y = jnp.meshgrid(
            jnp.arange(self.matrix.shape[0]), jnp.arange(self.matrix.shape[1]))
        return jnp.stack([x.reshape(-1), y.reshape(-1)]).T

    def split_iid(self, key, p=0.5):
        """Create Data split."""
        xy = self.grid()
        idx = random.permutation(key, jnp.arange(xy.shape[0]))

        train_size = int(xy.shape[0] * p)
        val_size = int((xy.shape[0] - train_size) / 2)
        return IndexSplit(
            train=xy[idx[:train_size]],
            val=xy[idx[train_size:train_size + val_size]],
            test=xy[idx[train_size + val_size:]])

    def split_kfold(self, key, k=2):
        """Perform k-fold minor data split into train, val, and test sets.

        The splits will have the following sizes:
          - train: floor(1/k)
          - val: floor((1 - train) / 2)
          - test: 1 - floor((1 - train) / 2)

        Assignments are exact, so each split will have exactly the same split
        sizes.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root PRNGKey for this operation.
        k : int
            Number of splits.
        """
        xy = self.grid()
        idx = random.permutation(key, jnp.arange(xy.shape[0]))
        size = int(xy.shape[0] / k)

        orders = xy[idx[(
            jnp.arange(xy.shape[0]).reshape(1, -1)
            + jnp.arange(k).reshape(-1, 1) * size
        ) % xy.shape[0]]]
        test_size = int((xy.shape[0] - size) / 2)

        return IndexSplit(
            train=orders[:, :size, :],
            val=orders[:, size:size + test_size, :],
            test=orders[:, size + test_size:, :])

    def split(self, key, splits=100, p=0.25):
        """Generate data splits using n sets of k-fold splits or IID splits.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root PRNGKey for this operation.
        splits : int
            Number of splits to generate.
        p : int
            Split proportion. If approximately 1/k, uses k-fold; otherwise,
            uses IID splits.
        """
        if p < 0.5 and np.abs(int(1 / p) - 1 / p) < 0.05:
            k = int(1 / p)
            n = int(splits / k)
            _, *keys = random.split(key, n + 1)
            splits = vmap(partial(self.split_kfold, k=k))(jnp.array(keys))
            return IndexSplit(
                train=splits.train.reshape(-1, *splits.train.shape[2:]),
                val=splits.val.reshape(-1, *splits.val.shape[2:]),
                test=splits.test.reshape(-1, *splits.test.shape[2:]))
        else:
            _, *keys = random.split(key, splits + 1)
            splits = vmap(partial(self.split_iid, p=p))(jnp.array(keys))
            return splits

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
