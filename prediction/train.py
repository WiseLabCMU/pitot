"""Training Loop."""

from collections import namedtuple
from functools import partial

from jax import numpy as jnp
from jax import random, jit, value_and_grad, vmap

import optax
import haiku as hk

from . import split
from .rank1 import Rank1
from .history import History


Result = namedtuple("Result", ["loss", "split", "pred", "baseline"])
Splits = namedtuple("Splits", ["train", "val", "test"])


class CrossValidationTrainer:
    """Training class for k-fold cross validation.

    Parameters
    ----------
    dataset : Dataset
        Source dataset, shared between replicates.
    model : callable() -> hk.Module
        Callable that creates the module to use.
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
    jit : bool
        Use jit on training step. Will be very slow if False -- should only
        disable jit for debugging!
    cpu : jaxlib.xla_extension.Device
        CPU to use to save data. If None, uses default.
    """

    def __init__(
            self, dataset, model, optimizer=None, multiplier=0.001,
            epochs=10, epoch_size=100, batch=64, jit=True, cpu=None):

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

        self.jit = jit
        self.cpu = cpu

    def train(self, key, train, val, test=None, base=None, tqdm=None):
        """Train model."""
        def _loss_func(params, x):
            pred = self.model.apply(params, x)
            if base is not None:
                pred = pred * self.multiplier + base[x[:, 0], x[:, 1]]
            return self.dataset.loss(pred, x)

        def _step(key, params, opt_state):
            batch = random.choice(key, train, axis=0, shape=(self.batch,))
            loss, grads = value_and_grad(
                _loss_func, allow_int=True)(params, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, loss, opt_state

        if self.jit:
            _step = jit(_step)

        _, kp, key = random.split(key, 3)
        params = self.model.init(kp, train[:1])
        opt_state = self.optimizer.init(params)

        iterator = range(self.epochs)
        if tqdm:
            iterator = tqdm(iterator)

        history = History(["train", "val", "test", "pred"], cpu=self.cpu)
        for _ in iterator:
            epoch_loss = []
            for _ in range(self.epoch_size):
                _, key, ks = random.split(key, 3)
                params, loss, opt_state = _step(ks, params, opt_state)
                epoch_loss.append(loss)

            pred = self.model.apply(params, None)
            if base is not None:
                pred = pred * self.multiplier + base
            history.log(
                train=jnp.mean(jnp.array(epoch_loss)),
                val=self.dataset.loss(pred, indices=val),
                test=self.dataset.loss(pred, indices=test),
                pred=pred)

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
            self, key=42, replicates=100, p=0.25, k=25, do_baseline=True,
            tqdm=None):
        """Train replicates.

        Parameters
        ----------
        key : jax.random.PRNGKey or int.
            Root random key; if int, creates one.
        replicates : int
            Number of replicates to train.
        p : float
            Target sparsity level (proportion of train+val set).
        k : int
            Number of folds for cross validation.
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
        offsets = jnp.arange(replicates) % self.dataset.shape[0]
        train, test =  vmap(partial(
            split.at_least_one, dim=self.dataset.shape,
            train=int(self.dataset.size * p)
        ))(split.keys(key, replicates), offsets)

        # Have to do this step outside because fit synchronizes globally
        if do_baseline:
            baseline = Rank1(self.dataset).fit_predict(train)
        else:
            baseline = jnp.zeros([replicates] + list(self.dataset.shape))

        # Generate train/val/test for full method
        train_final, val = vmap(partial(
            split.crossval, split=k
        ))(split.keys(key, train.shape[0]), train)

        # Inner k-fold replicates
        def _train(_key, train, val, test, baseline):
            return vmap(partial(
                self.train, base=baseline, test=test, tqdm=tqdm
            ))(split.keys(_key, train.shape[0]), train, val)

        # Outer IID replicates
        results = vmap(_train)(
            split.keys(key, train.shape[0]), train_final, val, test, baseline)

        # Final predictions
        pred_raw = jnp.mean(results["pred"], axis=1)
        best = jnp.argmin(jnp.mean(results["val"], axis=1), axis=1)
        pred = pred_raw[jnp.arange(replicates), best]

        # Generate results; must be single layer dictionary to be
        # compatible with np.savez.
        return {
            "train_loss": results["train"],
            "val_loss": results["val"],
            "test_loss": results["test"],
            "train_split": train_final,
            "val_split": val,
            "test_split": test,
            "predictions": pred,
            "baseline": baseline
        }
