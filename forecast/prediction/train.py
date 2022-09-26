"""Matrix Factorization Training."""

from collections import namedtuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, value_and_grad, vmap

import optax
import haiku as hk

from forecast.prediction.objective import MatrixFactorizationObjective

from . import split
from .rank1 import Rank1
from .history import History
from .objective import MatrixFactorizationObjective, InterferenceObjective


class CrossValidationTrainer:
    """Training class for k-fold cross validation.

    Parameters
    ----------
    dataset : Dataset
        Source dataset, shared between replicates.
    model : callable() -> hk.Module
        Callable that creates the model to use.
    optimizer : optax.GradientTransformation
        Optimizer to use.
    beta : (float, float)
        Weights applied to non-interference and interference loss,
        respectively.
    epochs : int
        Number of epochs (not a "true" epoch; just used for accounting).
        A checkpoint is saved (to main memory) after each epoch.
    epoch_size : int
        Number of batches per epoch; each batch is IID.
    batch : int or (int, int).
        Batch size; if tuple, sets the batch size for non-interference
        and interference separately.
    replicates : int
        Number of replicates to train.
    k : int
        Number of folds for cross validation.
    jit : bool
        Use jit on training step. Will be very slow if False -- should only
        disable jit for debugging!
    cpu : jaxlib.xla_extension.Device
        CPU to use to save data. If None, uses first CPU (jax.devices('cpu')).
    """

    TrainState = namedtuple("TrainState", ["params", "opt_state"])
    Replicate = namedtuple("Replicate", ["baseline", "splits_mf", "splits_if"])
    Splits = namedtuple("Splits", ["train", "val", "test"])
    Checkpoint = namedtuple(
        "Checkpoint", ["train", "val", "test", "pred", "module", "runtime"])

    def __init__(
            self, dataset, model, optimizer=None, beta=(1.0, 1.0),
            epochs=10, epoch_size=100, batch=64, replicates=100,
            k=25, jit=True, cpu=None):

        def _forward(x):
            return model()(x)

        self.model = hk.without_apply_rng(hk.transform(_forward))
        self.dataset = dataset
        self.optimizer = optax.adam(0.001) if optimizer is None else optimizer

        self.epochs = epochs
        self.epoch_size = epoch_size

        if isinstance(batch, int):
            batch = (batch, batch)
        self.obj_mf = MatrixFactorizationObjective(
            dataset, model, batch=batch[0], beta=beta[0])
        self.obj_if = InterferenceObjective(
            dataset, model, batch=batch[1], beta=beta[1])

        self.replicates = replicates
        self.k = k

        self.step = jax.jit(self._step) if jit else self._step
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

    def _init(self, key, train):
        """Initialize model parameters and optimization state."""
        params = self.model.init(key, train[:1])
        opt_state = self.optimizer.init(params)
        return self.TrainState(params=params, opt_state=opt_state)

    def _step(self, key, state, replicate):
        """Single training step."""
        def _loss_func(key, params, baseline, train_mf, train_if):
            k1, k2 = random.split(key, 2)
            return (
                self.obj_mf.sample_loss(k1, params, baseline, train_mf)
                + self.obj_if.sample_loss(k2, params, baseline, train_if))

        loss, grads = value_and_grad(_loss_func, allow_int=True)(
            key, state.params, replicate.baseline,
            replicate.splits_mf.train, replicate.splits_if.train)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, self.TrainState(params=params, opt_state=opt_state)

    def _epoch(self, key, state, replicate):
        """Single epoch for a single replicate."""
        epoch_loss = 0.
        for ki in random.split(key, self.epoch_size):
            loss, state = self.step(ki, state, replicate)
            epoch_loss += loss

        C_hat, M, D = self.model.apply(
            state.params, None, baseline=replicate.baseline)

        return self.Checkpoint(
            train=epoch_loss / self.epoch_size,
        )
