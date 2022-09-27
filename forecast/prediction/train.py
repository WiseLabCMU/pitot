"""Matrix Factorization Training."""

from collections import namedtuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, value_and_grad, vmap, jit

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
    cpu : jaxlib.xla_extension.Device
        CPU to use to save data. If None, uses first CPU (jax.devices('cpu')).
    """

    TrainState = namedtuple("TrainState", ["params", "opt_state"])
    Replicate = namedtuple(
        "Replicate", ["m_bar, d_bar", "splits_mf", "splits_if"])
    Splits = namedtuple("Splits", ["train", "val", "test"])

    def __init__(
            self, dataset, model, optimizer=None, beta=(1.0, 1.0), epochs=10,
            epoch_size=100, batch=64, replicates=100, k=25, cpu=None):

        def _forward(*args, **kwargs):
            return model()(*args, **kwargs)

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
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

    def _init(self, key, train):
        """Initialize model parameters and optimization state."""
        params = self.model.init(key, train[:1])
        opt_state = self.optimizer.init(params)
        return self.TrainState(params=params, opt_state=opt_state)

    @jit
    def step(self, key, state, replicate):
        """Single training step."""
        # Close over splits, m_bar, d_bar so they aren't included in
        # value_and_grad.
        def _loss_func(key, params):
            k1, k2 = random.split(key, 2)
            return (
                self.obj_mf.sample_loss(
                    k1, params, replicate.splits_mf.train,
                    m_bar=replicate.m_bar, d_bar=replicate.d_bar)
                + self.obj_if.sample_loss(
                    k2, params, replicate.splits_if.train,
                    m_bar=replicate.m_bar, d_bar=replicate.d_bar))

        loss, grads = value_and_grad(_loss_func)(key, state.params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, self.TrainState(params=params, opt_state=opt_state)

    def epoch_train(self, key, state, replicate):
        """Single epoch for a single replicate."""
        epoch_loss = 0.
        for ki in random.split(key, self.epoch_size):
            loss, state = self.step(ki, state, replicate)
            epoch_loss += loss
        return epoch_loss / self.epoch_size, state

    def epoch_val(self, state, replicate):
        """Handle epoch checkpointing."""
        checkpoint = self.model.apply(
            state.params,
            ij=self.dataset.if_ij, ip=self.dataset.if_ip,
            full=True, m_bar=replicate.m_bar, d_bar=replicate.d_bar)
        losses = {
            "if_val": self.obj_if(checkpoint, replicate.splits_if.val),
            "if_test": self.obj_if(checkpoint, replicate.splits_if.test),
            "mf_val": self.obj_mf(checkpoint, replicate.splits_mf.val),
            "mf_val": self.obj_mf(checkpoint, replicate.splits_mf.test),
        }
        return losses, checkpoint

    def train(self, key, replicate, tqdm=None):
        """Train model for a single swarm of k-CV replicates."""
        train_func = vmap(
            self.epoch_train, in_axes=(0, 0, self.Replicate(None, None, 0, 0)))
        val_func = vmap(
            self.epoch_val, in_axes=(0, self.Replicate(None, None, 0, 0)))

        k1, k2 = random.split(key, 2)
        state = vmap(self._init_)(split.keys(k1, self.k), replicate.if_train)

        iterator = random.split(k2, self.epochs)
        if tqdm is not None:
            iterator = tqdm(iterator)

        history = History(cpu=self.cpu)
        for ki in iterator:
            loss, state = train_func(split.keys(ki, self.k), state, replicate)
            losses, checkpoint = val_func(state, replicate)

            history.log(train=loss, **losses)
            history.update(**checkpoint)

        return history.export()

    def train_replicates(self, key=42, p=0.25, do_baseline=True, tqdm=None):
        """Train replicates.

        Parameters
        ----------
        key : jax.random.PRNGKey or int.
            Root random key; if int, creates one.
        p : float
            Target sparsity level (proportion of train+val set).
        do_baseline : bool
            Use baseline as starting point, and fit residuals only.
        tqdm : tqdm.tqdm or tqdm.notebook.tqdm
            Progress bar to use during training, if present.
        """
        # Create key
        if isinstance(key, int):
            key = random.PRNGKey(key)

        # Generate train/test
        train_mf, test_mf = self.obj_mf.split()

        # TODO TODO TODO

        # Have to do this step outside because fit synchronizes globally
        if do_baseline:
            _baseline = Rank1(self.dataset)
            m_bar, d_bar = _baseline.fit(train_mf)
            C_bar = _baseline.predict(m_bar, d_bar)
        else:
            x = jnp.zeros(self.dataset.shape[0])
            y = jnp.zeros(self.dataset.shape[1])

        # Generate train/val/test for full method
        train_mf, val_mf = vmap(partial(
            split.crossval, split=self.k
        ))(split.keys(key, train_mf.shape[0]), train_mf)

        # Todo: refactor split to include execution time
        # (operate over abstract data[])
