"""Training objectives."""

from collections import namedtuple
from functools import partial

from jax import random, vmap
from jax import numpy as jnp

from . import split


class MatrixFactorizationObjective:
    """Ordinary matrix factorization objective.

    Parameters
    ----------
    dataset : Dataset
        Source dataset.
    model : hk.Module
        Initialized haiku model.
    batch : int
        Batch size.
    beta : float
        Loss multiplier.
    """

    Data = namedtuple("MFData", ["ij", "C_ij"])

    def __init__(self, dataset, model, batch=64, beta=1.0):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch
        self.beta = beta

    def batch(self, key, data):
        """Generate IID data batch."""
        idx = random.randint(key, shape=(self.batch_size,))
        return self.Data(attr[idx] for attr in data)

    def loss(self, params, baseline, data):
        """Get objective loss."""
        pred = self.model.apply(params, data.ij, baseline=baseline)
        return jnp.mean(jnp.square(pred - data.C_ij)) * self.beta

    def loss_split(self, params, baseline, data):
        """Compute loss, splitting data into batches for memory."""
        batches = int(data.shape[0] / self.batch_size)

        loss = 0.
        for idx in jnp.arange(batches):
            loss += self.loss(
                params, baseline, self.Data(
                    attr[idx:idx + self.batch_size] for attr in self.data))
        return loss / batches

    def sample_loss(self, key, params, baseline, data):
        """Generate batch and compute loss."""
        batch = self.batch(key, data)
        return self.loss(params, baseline, batch)

    def split(self, key, p=0.5, replicates=32):
        """Generate train/test splits."""
        offsets = jnp.arange(self.replicates) % self.dataset.shape[0]
        return vmap(partial(
            split.at_least_one, dim=self.dataset.shape,
            train=int(self.dataset.size * p - self.dataset.shape[0])
        ))(split.keys(key, replicates), offset=offsets)



class InterferenceObjective(MatrixFactorizationObjective):
    """Matrix factorization interference objective."""

    Data = namedtuple("IFData", ["ij", "ip", "C_ij_ip"])

    def loss(self, params, baseline, data):
        """Get objective loss."""
        pred = self.model.apply(params, data.ij, ip=data.ip, baseline=baseline)
        return jnp.mean(jnp.square(pred - data.C_ij_ip)) * self.beta

    def split(self, key, p=0.5, replicates=32):
        """Generate train/val/test splits."""
        train, test = vmap(partial(
            split.crossval, split=self.k
        ))(split.keys(key, replicates), )
