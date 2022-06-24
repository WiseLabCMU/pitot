"""Training Loop."""

from collections import namedtuple

from jax import numpy as jnp
from jax import random, jit, value_and_grad, vmap

import optax
import haiku as hk


Result = namedtuple("Result", ["train", "val", "test", "splits", "params"])


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
        Number of epochs.
    epoch_size : int
        Number of batches per epoch; each batch is IID.
    batch : int
        Batch size.
    tqdm : tqdm.tqdm or tqdm.notebook.tqdm
        Progress bar to use during training, if present.
    """

    def __init__(
            self, dataset, model, optimizer=None,
            epochs=100, epoch_size=100, batch=64, tqdm=None):

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
        self.tqdm = tqdm

    def train(self, key, splits):
        """Train model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root random key for this training loop.
        splits : IndexSplit
            train, val, test indices.

        Returns
        -------
        Result
            Losses by epoch and parameters; also includes splits.
        """

        @jit
        def _loss_func(params, x):
            return self.dataset.loss(self.model.apply(params, x), x)

        @jit
        def _step(key, params, opt_state):
            batch = random.choice(
                key, splits.train, axis=0, shape=(self.batch,))
            loss, grads = value_and_grad(
                _loss_func, allow_int=True)(params, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            return params, loss, opt_state

        _, kp, key = random.split(key, 3)
        params = self.model.init(kp, splits.train[:1])
        opt_state = self.optimizer.init(params)

        if self.tqdm:
            iterator = self.tqdm(range(self.epochs))
        else:
            iterator = range(self.epochs)

        train = []
        val = []
        test = []
        for _ in iterator:
            epoch_loss = []
            for _ in range(self.epoch_size):
                _, key, ks = random.split(key, 3)
                params, loss, opt_state = _step(ks, params, opt_state)
                epoch_loss.append(loss)

            train.append(jnp.mean(jnp.array(epoch_loss)))
            val.append(_loss_func(params, splits.val))
            test.append(_loss_func(params, splits.test))

        return Result(
            train=jnp.array(train), val=jnp.array(val), test=jnp.array(test),
            splits=splits, params=params)

    def predictions(self, params):
        """Generate prediction matrix for parameters."""
        xy = self.dataset.grid()
        pred = self.model.apply(params, xy)
        return jnp.zeros_like(
            self.dataset.matrix).at[xy[:, 0], xy[:, 1]].set(pred)

    def export_results(self, results):
        """Create dictionary of results for saving."""
        return {
            "train": jnp.sqrt(results.train),
            "val": jnp.sqrt(results.val),
            "test": jnp.sqrt(results.test),
            "split_train": vmap(self.dataset.to_mask)(results.splits.train),
            "split_val": vmap(self.dataset.to_mask)(results.splits.val),
            "split_test": vmap(self.dataset.to_mask)(results.splits.test),
            "predictions": vmap(self.predictions)(results.params)
        }

    def train_splits(self, key, splits):
        """Train cross-validation splits."""
        _, *keys = random.split(key, splits.train.shape[0] + 1)
        return vmap(self.train)(jnp.array(keys), splits)

    def train_replicates(self, key, replicates=100, p=0.25, k=24):
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

        Returns
        -------
        Result
            Losses by epoch and parameters; also includes splits.
        """
        _, ks, kt = random.split(key, 3)
        splits = self.dataset.split(ks, splits=replicates, p=p, kval=k)
        _, *keys = random.split(kt, splits.train.shape[0] + 1)
        return vmap(self.train_splits)(jnp.array(keys), splits)
