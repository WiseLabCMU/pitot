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

    def batch(self, key, indices):
        """Generate IID data batch.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key to use.
        indices : jnp.array(int[:, 2])
            Train split (i, j) indices.
        """
        batch = random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=indices.shape[0])
        return self.Data(
            ij=indices[batch], C_ij=self.dataset.index(indices[batch]))

    def loss(self, params, data, **kwargs):
        """Get objective loss.

        Parameters
        ----------
        params : pytree
            Model parameters.
        data : self.Data
            Named tuple with ij and C_ij attributes.
        kwargs : dict
            Additional parameters passed through to the model.

        Returns
        -------
        float
            Loss value (MSE) weighted by the objective multiplier (beta).
        """
        pred = self.model.apply(params, data.ij, **kwargs)
        return jnp.mean(jnp.square(pred - data.C_ij)) * self.beta

    def sample_loss(self, key, params, indices, **kwargs):
        """Generate batch and compute loss."""
        batch = self.batch(key, indices)
        return self.loss(params, batch, **kwargs)

    def split(self, key, p=0.5, replicates=32):
        """Generate train/test split indices."""
        offsets = jnp.arange(self.replicates) % self.dataset.shape[0]
        return vmap(partial(
            split.at_least_one, dim=self.dataset.shape,
            train=int(self.dataset.size * p - self.dataset.shape[0])
        ))(split.keys(key, replicates), offset=offsets)

    def val_loss(self, pred, idx):
        """Validation loss with predictions from _predict_full."""
        return jnp.mean(jnp.square(
            pred[idx[:, 0], idx[:, 1]]
            - self.dataset.matrix[idx[:, 0], idx[:, 1]]))


class InterferenceObjective(MatrixFactorizationObjective):
    """Matrix factorization interference objective."""

    Data = namedtuple("IFData", ["ij", "ip", "C_ij_ip"])

    def batch(self, key, indices):
        """Generate IID data batch."""
        batch = indices[random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=indices.shape[0])]
        return self.Data(
            ij=self.dataset.if_ij[batch],
            ip=self.dataset.if_ip[batch],
            C_ij_ip=self.dataset.interference[batch])

    def loss(self, params, data, **kwargs):
        """Get objective loss."""
        pred = self.model.apply(params, data.ij, ip=data.ip, **kwargs)
        return jnp.mean(jnp.square(pred - data.C_ij_ip)) * self.beta

    def split(self, key, p=0.5, replicates=32):
        """Generate train/val/test split indices."""
        return vmap(partial(
            split.iid, train=int(self.dataset.if_size * p), data=None
        ))(split.keys(key, replicates))

    def val_loss(self, pred, idx):
        """Validation loss with predictions from _predict_full."""
        return jnp.mean(jnp.square(
            pred["C_ij_ip_hat"][idx] - self.dataset.interference[idx]))
