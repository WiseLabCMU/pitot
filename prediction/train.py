"""Training Loop."""

from collections import namedtuple
from functools import partial

from jax import numpy as jnp
from jax import random, jit, value_and_grad, vmap

import optax
import haiku as hk

from . import split
from .rank1 import Rank1


Result = namedtuple("Result", ["loss", "splits", "params", "base"])
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
    epochs : int
        Number of epochs (not a "true" epoch; just used for accounting).
    epoch_size : int
        Number of batches per epoch; each batch is IID.
    batch : int
        Batch size.
    logsumexp: bool
        Use logsumexp instead of simple sum.
    """

    def __init__(
            self, dataset, model, optimizer=None,
            epochs=100, epoch_size=100, batch=64):

        def _forward(x):
            return model()(x)

        self.model = hk.without_apply_rng(hk.transform(_forward))
        self.dataset = dataset
        if optimizer is None:
            optimizer = optax.adam(0.001)
        self.optimizer = optimizer

        self.epochs = epochs
        self.epoch_size = epoch_size
        self.batch = batch

    def train(self, key, train, val, test=None, base=None, tqdm=None):
        """Train model."""
        @jit
        def _loss_func(params, x):
            pred = self.model.apply(params, x) + base[x[:, 0], x[:, 1]]
            return self.dataset.loss(pred, x)

        @jit
        def _step(key, params, opt_state):
            batch = random.choice(key, train, axis=0, shape=(self.batch,))
            loss, grads = value_and_grad(
                _loss_func, allow_int=True)(params, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, loss, opt_state

        _, kp, key = random.split(key, 3)
        params = self.model.init(kp, train[:1])
        opt_state = self.optimizer.init(params)

        iterator = range(self.epochs)
        if tqdm:
            iterator = tqdm(iterator)

        losses = Splits([], [], [])
        for _ in iterator:
            epoch_loss = []
            for _ in range(self.epoch_size):
                _, key, ks = random.split(key, 3)
                params, loss, opt_state = _step(ks, params, opt_state)
                epoch_loss.append(loss)

            losses.train.append(jnp.mean(jnp.array(epoch_loss)))
            losses.val.append(_loss_func(params, val))
            losses.test.append(_loss_func(params, test))

        return Result(
            loss=Splits(*(jnp.array(x) for x in losses)),
            splits=Splits(train, val, test),
            params=params, base=base)

    def predictions(self, params, base=None):
        """Generate prediction matrix for parameters."""
        xy = self.dataset.grid()
        pred = self.model.apply(params, xy)
        res = jnp.zeros_like(
            self.dataset.matrix).at[xy[:, 0], xy[:, 1]].set(pred)
        if base is not None:
            res += base
        return res

    def export_results(self, results):
        """Create dictionary of results for saving."""
        return {
            "train": jnp.sqrt(results.train),
            "val": jnp.sqrt(results.val),
            "test": jnp.sqrt(results.test),
            "split_train": vmap(self.dataset.to_mask)(results.splits.train),
            "split_val": vmap(self.dataset.to_mask)(results.splits.val),
            "split_test": vmap(self.dataset.to_mask)(results.splits.test),
            "predictions": vmap(self.predictions)(results.params, results.base)
        }

    def train_replicates(self, key, replicates=100, p=0.25, k=25, tqdm=None):
        """Train replicates.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root random key.
        replicates : int
            Number of replicates to train.
        p : float
            Target sparsity level (proportion of train+val set).
        k : int
            Number of folds for cross validation.
        tqdm : tqdm.tqdm or tqdm.notebook.tqdm
            Progress bar to use during training, if present.

        Returns
        -------
        Result
            Losses by epoch and parameters; also includes splits.
        """
        # Generate train/test
        offsets = jnp.arange(replicates) % self.dataset.shape[0]
        train, test = vmap(partial(
            split.at_least_one, dim=self.dataset.shape,
            train=int(self.dataset.size * p)
        ))(split.keys(key, replicates), offsets)

        # Have to do this step outside because fit synchronizes globally
        base = Rank1(self.dataset).fit_predict(train)

        # Generate train/val/test for full method
        train_final, val = vmap(partial(
            split.crossval, split=k
        ))(split.keys(key, train.shape[0]), train)

        # Inner k-fold replicates
        def _train(_key, train, val, test, base):
            return vmap(partial(
                self.train, base=base, test=test, tqdm=tqdm
            ))(split.keys(_key, train.shape[0]), train, val)

        # Outer IID replicates
        return vmap(_train)(
            split.keys(key, train.shape[0]), train_final, val, test, base)
