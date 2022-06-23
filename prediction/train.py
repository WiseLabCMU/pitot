"""Training loop."""

from collections import namedtuple

import numpy as np
from jax import numpy as jnp
from jax import random, jit, value_and_grad, vmap

import optax
import haiku as hk


Result = namedtuple("Result", ["train", "val", "test", "params", "splits"])


def vmap2(x):
    """2-level vmap helper."""
    return vmap(vmap(x))


class ReplicateTrainer:
    """Training class for multiple replicates; captures shared objects.

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
            params=params, splits=splits)

    def train_replicates(self, key, replicates=100, p=0.25):
        """Train replicates.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Root random key.
        replicates : int
            Number of replicates to train.
        p : float
            Target sparsity level (proportion of train set).

        Returns
        -------
        Result
            Losses by epoch and parameters; also includes splits.
        """
        _, ks, key = random.split(key, 3)
        splits = self.dataset.split(ks, splits=replicates, p=p)
        _, *keys = random.split(key, replicates + 1)
        return vmap2(self.train)(jnp.array(keys), splits)

    def predictions(self, params):
        """Generate prediction matrix for parameters."""
        xy = self.dataset.grid()
        pred = self.model.apply(params, xy)
        return jnp.zeros_like(
            self.dataset.matrix).at[xy[:, 0], xy[:, 1]].set(pred)

    def save_results(self, results, file="results.npz"):
        """Save results (and predictions) to disk."""
        np.savez_compressed(file, **{
            "train": results.train,
            "val": results.val,
            "test": results.test,
            "split_train": vmap2(self.dataset.to_mask)(results.splits.train),
            "split_val": vmap2(self.dataset.to_mask)(results.splits.val),
            "split_test": vmap2(self.dataset.to_mask)(results.splits.test),
            "pred": vmap2(self.predictions)(results.params)
        })
