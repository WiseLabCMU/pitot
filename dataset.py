"""Dataset for prediction."""

import numpy as np
import torch


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
            self, data="data/polybench/20-40.npz", device=None,
            val=0.5, test=0.0):

        if isinstance(data, str):
            data = dict(np.load(data))
        self.data = data
        self.matrix = np.log(data['runtime'])
        self.rms = np.sqrt(np.mean(np.square(self.matrix)))
        self.matrix_torch = torch.tensor(self.matrix)
        if device is not None:
            self.matrix_torch = self.matrix_torch.to(device)

        self.opcodes = np.log(data['opcodes'].astype(np.float32) + 1)
        self.files = data['files']
        self.runtimes = data['runtimes']
        self.size = np.sum(data['runtime'] > 0)

        self._split(val, test)

    def _split(self, val=0.5, test=0.):

        x, y = np.meshgrid(
            np.arange(self.matrix.shape[0]), np.arange(self.matrix.shape[1]))
        xy = np.stack([x.reshape(-1), y.reshape(-1)]).T
        idx = np.arange(xy.shape[0])
        np.random.shuffle(idx)

        val_size = int(val * self.size)
        test_size = int(test * self.size)
        train_size = self.size - val_size - test_size

        self.splits = {
            'train': xy[idx[:train_size]],
            'val': xy[idx[train_size: train_size + val_size]],
            'test': xy[idx[train_size + val_size:]]
        }

    def loss(self, pred, idx=None, split='val'):
        """Compute loss."""
        if idx is None:
            idx = self.splits[split]
        y_true = self.matrix_torch[idx[:, 0], idx[:, 1]]
        return torch.mean(torch.square(pred - y_true))

    def sample(self, batch=64, split='train'):
        """Get split of indices."""
        split = self.splits[split]
        choices = np.random.choice(np.arange(split.shape[0]), size=batch)
        return split[choices]
