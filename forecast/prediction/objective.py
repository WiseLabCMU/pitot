"""Training objectives."""

from collections import namedtuple

from jax import random
from jax import numpy as jnp


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

    def sample_loss(self, key, params, baseline, data):
        """Generate batch and compute loss."""
        batch = self.batch(key, data)
        return self.loss(params, baseline, batch)


class InterferenceObjective(MatrixFactorizationObjective):
    """Matrix factorization interference objective."""

    Data = namedtuple("IFData", ["ij", "ip", "C_ij_ip"])

    def loss(self, params, baseline, data):
        """Get objective loss."""
        pred = self.model.apply(params, data.ij, ip=data.ip, baseline=baseline)
        return jnp.mean(jnp.square(pred - data.C_ij_ip)) * self.beta
