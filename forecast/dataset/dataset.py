"""Dataset for prediction."""

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt


class Dataset:
    """Matrix Factorization Dataset.

    Parameters
    ----------
    data : str or dict
        Source data, or filepath to npz file containing data.
    if_data : str or dict
        Source interference data or filepath.
    key : callable or str
        Name of key or callable that fetches the target value from the archive.
        Should be the same key that was used to make if_data.
    offset : float
        Fixed offset to apply to data (for scale normalization)

    Attributes
    ----------
    See writeup.

    matrix : jnp.array(float[N_m, N_d])
        Non-interference data matrix.
    interference : jnp.array(float[])
        Execution time for interference data.
    N_m, N_d : int
        Number of modules, devices.
    x_m, x_d : jnp.array(float[n_features, N_m or N_d])
        Side information for modules, devices.
    if_ijk : jnp.array(int[:, 3])
        Module, device, and interference indices (i, j, k).
    if_if : jnp.array(int[])
        Interference data.
    """

    def __init__(
            self, data="data.npz", if_data=None,
            key=lambda x: x['mean'], offset=1.):

        # Matrix
        self.data = self._load(data)
        self.matrix = jnp.log(self._get_key(self.data, key)) - jnp.log(offset)
        self.shape = self.matrix.shape
        self.N_m, self.N_d = self.shape
        self.size = np.prod(self.shape)

        # Side information
        self.x_m = jnp.log(self.data['module_data'].astype(jnp.float32) + 1)
        self.x_d = jnp.array(self.data['runtime_data'])

        # Interference
        if if_data is not None:
            self.if_data = self._load(if_data)
            self.if_ijk = jnp.stack([
                self.if_data["module"],
                self.if_data["runtime"],
                self.if_data["interferer"]]).T
            self.interference = jnp.log(
                self._get_key(self.if_data, key)) - jnp.log(offset)
            self.if_size = self.interference.shape[0]
        else:
            self.if_data = None
            self.if_ijk = None
            self.interference = None
            self.if_size = 0

        # Metadata
        self.modules = self.data['modules']
        self.modules_dict = {m: i for i, m in enumerate(self.modules)}
        self.runtimes = self.data['runtimes']
        self.runtimes_dict = {r: i for i, r in enumerate(self.runtimes)}
        self.cpu_names = self.data['cpu_names']
        self.opcode_names = self.data['opcode_names']

    @staticmethod
    def _load(path):
        if isinstance(path, str):
            return dict(np.load(path))
        else:
            return path

    @staticmethod
    def _get_key(data, key):
        if callable(key):
            return key(data)
        elif isinstance(key, str):
            return data[key]
        else:
            raise TypeError(
                "Key {} is not callable or string index.".format(key))

    def grid(self):
        """Create (x, y) grid pairs."""
        x, y = jnp.meshgrid(np.arange(self.N_m), jnp.arange(self.N_d))
        return jnp.stack([x.reshape(-1), y.reshape(-1)]).T

    def to_mask(self, xy):
        """Convert indices to mask."""
        return jnp.zeros_like(self.matrix).at[xy[:, 0], xy[:, 1]].set(1.)

    def index(self, indices):
        """Index into data."""
        return self.matrix[indices[:, 0], indices[:, 1]]

    def index_if(self, indices):
        """Index into interference data."""
        if self.if_data is None:
            return None
        else:
            return self.if_ijk[indices]

    def _index(self, pred, indices=None, mode="mf"):
        if mode == "mf":
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
        elif mode == "if":
            if indices is not None:
                return pred, self.interference[indices]
            else:
                return pred, self.interference

    def loss(self, pred, indices=None, mode="mf"):
        """Compute squared l2-log loss.

        Parameters
        ----------
        pred : jnp.array
            Predictions. Can be full matrix, or a sparse list.
        indices : jnp.array
            Prediction indices. If None, pred is treated as a full matrix.
        mode : str
            Loss mode; can be "mf" (Matrix Factorization) or "if"
            (Interference).
        """
        if (mode == "if" and self.if_data is None) or pred.shape[0] == 0:
            return 0.0
        pred, actual = self._index(pred, indices=indices, mode=mode)
        return jnp.mean(jnp.square(pred - actual))

    def rmse(self, pred, indices=None, mode="mf"):
        """Compute RMSE log error; alias for sqrt(loss)."""
        return jnp.sqrt(self.loss(pred=pred, indices=indices, mode=mode))

    def error(self, pred, indices=None, mode="mf", full=False):
        """Compute mean absolute error."""
        pred, actual = self._index(pred, indices=indices, mode=mode)
        if full:
            return pred - actual
        else:
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

        return ax
