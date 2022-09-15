"""Dataset for prediction."""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


IndexSplit = namedtuple("IndexSplit", ['train', 'test'])


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
    offset : float
        Fixed offset to apply to data (for scale normalization)
    """

    def __init__(
            self, data="data.npz", key=lambda x: x['mean'], device=None,
            offset=1.):

        # Data
        if isinstance(data, str):
            data = dict(np.load(data))
        self.data = data

        # Matrix
        self._data_key = key(data)
        self.matrix = np.log(self._data_key) - np.log(offset)
        self.shape = self.matrix.shape
        self.size = np.prod(self.shape)

        # Side information
        self.module_data = jnp.log(data['module_data'].astype(jnp.float32) + 1)
        self.module_data_pca = jnp.array(PCA().fit_transform(self.module_data))
        self.runtime_data = jnp.array(data['runtime_data'])

        # Send to GPU if applicable
        if device is not None:
            self._matrix = self.matrix.to(device)
            self._module_data = self.module_data.to(device)
            self._runtime_data = self.runtime_data.to(device)

        # Metadata
        self.modules = data['modules']
        self.modules_dict = {m: i for i, m in enumerate(self.modules)}
        self.runtimes = data['runtimes']
        self.runtimes_dict = {r: i for i, r in enumerate(self.runtimes)}
        self.cpu_names = data['cpu_names']
        self.opcode_names = data['opcode_names']

    def grid(self):
        """Create (x, y) grid pairs."""
        x, y = jnp.meshgrid(
            np.arange(self.shape[0]), jnp.arange(self.shape[1]))
        return jnp.stack([x.reshape(-1), y.reshape(-1)]).T

    def to_mask(self, xy):
        """Convert indices to mask."""
        return jnp.zeros_like(self.matrix).at[xy[:, 0], xy[:, 1]].set(1.)

    def index(self, indices):
        """Index into data."""
        return self.matrix[indices[:, 0], indices[:, 1]]

    def _index(self, pred, indices=None):
        # Indices available -> index into matrix
        if indices is not None:
            actual = self.index(indices)
            # Full matrix predictions -> also index given predictions
            if len(pred.shape) == 2:
                pred = pred[indices[:, 0], indices[:, 1]]
        # No indices -> assume full matrix
        else:
            assert len(pred.shape) == 2
            actual = self.matrix
        return pred, actual

    def loss(self, pred, indices=None):
        """Compute squared l2-log loss.

        Parameters
        ----------
        pred : jnp.array
            Predictions. Can be full matrix, or a sparse list.
        indices : jnp.array
            Prediction indices. If None, pred is treated as a full matrix.
        """
        pred, actual = self._index(pred, indices=indices)
        return jnp.mean(jnp.square(pred - actual))

    def rmse(self, pred, indices=None):
        """Compute RMSE log error; alias for sqrt(loss)."""
        return jnp.sqrt(self.loss(pred=pred, indices=indices))

    def errors(self, pred, indices=None):
        """Get all errors."""
        pred, actual = self._index(pred, indices=indices)
        return pred - actual

    def error(self, pred, indices=None):
        """Compute mean absolute error."""
        pred, actual = self._index(pred, indices=indices)
        return jnp.mean(jnp.abs(pred - actual))

    def plot(self, matrix=None, ax=None, figsize=None, title=None):
        """Plot results."""
        if matrix is None:
            matrix = self.matrix
        if ax is None:
            if figsize is None:
                figsize = (matrix.shape[0] * 0.2, matrix.shape[1] * 0.2)
            _, ax = plt.subplots(1, 1, figsize=figsize)
        if title:
            ax.set_ylabel(title)

        ax.imshow(matrix.T)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(self.runtimes)))
        ax.set_yticklabels(self.runtimes)
