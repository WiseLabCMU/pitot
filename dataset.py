"""Dataset for prediction."""

import numpy as np
import torch


class Dataset:

    def __init__(self, source="data/polybench/20-40.npz"):

        data = dict(np.load(source))
        self.matrix = data['runtime']
        self.opcodes = data['opcodes'].astype(np.float32)

        self.files = data['files']
        self.runtimes = data['runtimes']

        self.size = np.sum(data['runtime'] > 0)

    def partition(self, ratio=0.5):
        """Partition dataset into train and test sets."""

        x, y = np.meshgrid(
            np.arange(self.matrix.shape[0]), np.arange(self.matrix.shape[1]))
        xy = np.stack([x.reshape(-1), y.reshape(-1)]).T
        idx = np.arange(xy.shape[0])
        np.random.shuffle(idx)

        train = int(ratio * self.size)
        idxtrain = idx[:train]
        idxval = idx[train:]

        return xy[idxtrain], xy[idxval]
