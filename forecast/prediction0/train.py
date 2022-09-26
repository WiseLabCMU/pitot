"""Training Loop."""

from collections import namedtuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, value_and_grad, vmap

import optax
import haiku as hk

from . import split
from ..prediction.rank1 import Rank1
from .history import History


Checkpoint = namedtuple(
    "Checkpoint", ["train", "val", "test", "pred", "module", "runtime"])


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
    multiplier : float
        Multiplier to apply to residual fitting.
    epochs : int
        Number of epochs (not a "true" epoch; just used for accounting).
        A checkpoint is saved (to main memory) after each epoch.
    epoch_size : int
        Number of batches per epoch; each batch is IID.
    batch : int
        Batch size.
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

    def __init__(
            self, dataset, model, optimizer=None, multiplier=0.001,
            epochs=10, epoch_size=100, batch=64, replicates=100, k=25,
            jit=True, cpu=None):

        def _forward(x):
            return model()(x)

        self.model = hk.without_apply_rng(hk.transform(_forward))
        self.dataset = dataset
        if optimizer is None:
            optimizer = optax.adam(0.001)
        self.optimizer = optimizer
        self.multiplier = multiplier

        self.epochs = epochs
        self.epoch_size = epoch_size
        self.batch = batch
        self.replicates = replicates
        self.k = k

        self.step = jax.jit(self._step) if jit else self._step
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

    def _init(self, key, train):
        """Initialize model parameters and optimization state."""
        params = self.model.init(key, train[:1])
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def _step(self, key, params, opt_state, train, baseline):
        """Single training step."""
        def _loss_func(params, x):
            pred = (
                self.model.apply(params, x) * self.multiplier
                + baseline[x[:, 0], x[:, 1]])
            return self.dataset.loss(pred, x)

        batch = random.choice(key, train, axis=0, shape=(self.batch,))
        loss, grads = value_and_grad(
            _loss_func, allow_int=True)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def _epoch(self, key, params, opt, train, val, test=None, baseline=None):
        """Single epoch for a single replicate."""
        epoch_loss = 0.
        for ki in random.split(key, self.epoch_size):
            loss, params, opt = self.step(ki, params, opt, train, baseline)
            epoch_loss += loss

        pred, module_emb, runtime_emb = self.model.apply(params, None)
        pred = pred * self.multiplier + baseline

        return Checkpoint(
            train=epoch_loss / self.epoch_size,
            val=self.dataset.loss(pred, indices=val),
            test=self.dataset.loss(pred, indices=test),
            pred=pred, module=module_emb, runtime=runtime_emb), params, opt

    def train(self, key, train, val, test, baseline, tqdm=None):
        """Train model for a single swarm of k-CV replicates."""
        epoch_func = vmap(partial(self._epoch, test=test, baseline=baseline))

        k1, k2 = random.split(key, 2)

        params, opt_state = vmap(self._init)(split.keys(k1, self.k), train)

        iterator = random.split(k2, self.epochs)
        if tqdm:
            iterator = tqdm(iterator)

        history = History(cpu=self.cpu)
        for ki in iterator:
            epoch, params, opt_state = epoch_func(
                split.keys(ki, self.k), params, opt_state, train, val)

            history.log(
                train_loss=epoch.train, val_loss=epoch.val,
                test_loss=epoch.test)
            history.update(
                jnp.mean(epoch.val), predictions=epoch.pred,
                module_emb=epoch.module, runtime_emb=epoch.runtime)
        return history.export()

    def predictions(self, params, base=None):
        """Generate prediction matrix for parameters."""
        xy = self.dataset.grid()
        pred = self.model.apply(params, xy)
        res = jnp.zeros_like(
            self.dataset.matrix).at[xy[:, 0], xy[:, 1]].set(pred)
        if base is not None:
            res += base
        return res

    def train_replicates(
            self, key=42, p=0.25, do_baseline=True,
            tqdm=None):
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
              - train_loss, val_loss, test_loss: (replicates, k, epochs)
              - train_split, val_split: (replicates, k, samples, 2)
              - test_split: (replicates, samples, 2)
              - pred: (replicates, epochs, modules, runtimes)
              - baseline: (replicates, modules, runtimes)
        """
        # Create key
        if isinstance(key, int):
            key = random.PRNGKey(key)

        # Generate train/test
        offsets = jnp.arange(self.replicates) % self.dataset.shape[0]
        train, test = vmap(partial(
            split.at_least_one, dim=self.dataset.shape,
            train=int(self.dataset.size * p - self.dataset.shape[0])
        ))(split.keys(key, self.replicates), offsets)

        # Have to do this step outside because fit synchronizes globally
        if do_baseline:
            baseline = Rank1(self.dataset).fit_predict(train)
        else:
            baseline = jnp.zeros([self.replicates] + list(self.dataset.shape))

        # Generate train/val/test for full method
        train_final, val = vmap(partial(
            split.crossval, split=self.k
        ))(split.keys(key, train.shape[0]), train)

        results = vmap(
            partial(self.train, tqdm=tqdm)
        )(split.keys(key, self.replicates), train_final, val, test, baseline)

        # Generate results; must be single layer dictionary to be
        # compatible with np.savez.
        return {
            "train_split": train_final,
            "val_split": val,
            "test_split": test,
            "baseline": baseline,
            **results
        }
