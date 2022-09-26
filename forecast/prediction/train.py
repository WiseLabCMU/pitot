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
        "Checkpoint",
        ["train", "losses", "C_hat", "M", "D"])

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
        self.val = jax.jit(self._val) if jit self.self._val
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

    def _val(self, C_hat, state, replicate):
        """Validation losses."""
        return {
            "mf_val": self.dataset.loss(
                C_hat, indices=replicate.splits_mf.val),
            "mf_test": self.dataset.loss(
                C_hat, indices=replicate.splits_mf.test),
            "if_val": self.obj_mf.loss_split(
                state.params, replicate.baseline, replicate.splits_mf.val),
            "if_test": self.obj_mf.loss_split(
                state.params, replicate.baseline, replicate.splits_if.test)
        }

    def _epoch(self, key, state, replicate):
        """Single epoch for a single replicate."""
        epoch_loss = 0.
        for ki in random.split(key, self.epoch_size):
            loss, state = self.step(ki, state, replicate)
            epoch_loss += loss

        C_hat, M, D = self.model.apply(
            state.params, None, baseline=replicate.baseline)
        losses = self.val(C_hat, state, replicate)
        return state, self.Checkpoint(
            train=epoch_loss / self.epoch_size, losses=losses,
            C_hat=C_hat, M=M, D=D)

    def train(self, key, replicate, tqdm=None):
        """Train model for a single swarm of k-CV replicates."""
        epoch_func = vmap(
            self._epoch, in_axes=(0, 0, self.Replicate(None, 0, 0)))

        k1, k2 = random.split(key, 2)
        state = vmap(self._init_)(split.keys(k1, self.k), replicate.if_train)

        iterator = random.split(k2, self.epochs)
        if tqdm is not None:
            iterator = tqdm(iterator)

        history = History(cpu=self.cpu)
        for ki in iterator:
            state, epoch = epoch_func(split.keys(ki, self.k), state, replicate)

            history.log(train=epoch.train, **epoch.losses)
            history.update(
                jnp.mean(epoch.losses["if_val"] + epoch.losses("mf_val")),
                C_hat=epoch.C_hat, M=epoch.M, D=epoch.D)

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

        Returns
        -------
        dict
            Losses by epoch and parameters; also includes splits.
            Keys with shape:
            - train, mf_val, mf_test, if_val, mf_test: (replicates, k, epochs)
            - train_split, val_split: (replicates, k, samples, 2)
            - test_split: (replicates, samples, 2)
            - pred: (replicates, epochs, modules, runtimes)
            - baseline: (replicates, modules, runtimes)
        """
        # Create key
        if isinstance(key, int):
            key = random.PRNGKey(key)

        # Generate train/test
        train_mf, test_mf = self.obj_mf.split()

        # Have to do this step outside because fit synchronizes globally
        if do_baseline:
            baseline = Rank1(self.dataset).fit_predict(train_mf)
        else:
            baseline = jnp.zeros([self.replicates] + list(self.dataset.shape))

        # Generate train/val/test for full method
        train_mf, val_mf = vmap(partial(
            split.crossval, split=self.k
        ))(split.keys(key, train_mf.shape[0]), train_mf)

        # Todo: refactor split to include execution time
        # (operate over abstract data[])
