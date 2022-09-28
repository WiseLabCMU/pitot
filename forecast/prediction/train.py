"""Matrix Factorization Training."""

from collections import namedtuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, value_and_grad, vmap, jit

import optax
import haiku as hk


from . import split
from .rank1 import Rank1
from .history import History


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
        self.batch = batch
        self.beta = beta

        self.replicates = replicates
        self.k = k
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

        self.Splits_vmap = self.Splits(train=0, val=0, test=None)
        self.Replicate_vmap = self.Replicate(
            m_bar=None, d_bar=None,
            splits_mf=self.Splits_vmap, splits_if=self.Splits_vmap)

    def _init(self, key, train):
        """Initialize model parameters and optimization state."""
        params = self.model.init(key, train[:1])
        opt_state = self.optimizer.init(params)
        return self.TrainState(params=params, opt_state=opt_state)

    @jit
    def step(self, key, state, replicate):
        """Single training step."""
        k1, k2 = random.split(key, 2)
        ijk_mf = split.batch(k1, replicate.splits_mf, batch=self.batch[0])
        idx_if = split.batch(k2, replicate.splits_if, batch=self.batch[1])
        ijk_if = self.dataset.if_ijk[idx_if]

        # Close over all but params so they aren't included in value_and_grad.
        def _loss_func(params):
            pred_mf, pred_if = self.model.apply(
                params, [ijk_mf, ijk_if],
                m_bar=replicate.m_bar, d_bar=replicate.d_bar)
            return (
                self.dataset.loss(pred_mf, ijk_mf, mode="mf") * self.beta[0]
                + self.dataset.loss(pred_if, idx_if, mode="if") * self.beta[1])

        loss, grads = value_and_grad(_loss_func)(state.params)
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
        val_test = [
            self.dataset.if_ijk[replicate.splits_if.val],
            self.dataset.if_ijk[replicate.splits_if.test]]

        (if_val, if_test), checkpoint = self.model.apply(
            state.params, val_test, full=True,
            m_bar=replicate.m_bar, d_bar=replicate.d_bar)

        losses = {
            "mf_val_loss": self.dataset.loss(
                checkpoint["C_hat"], indices=replicate.splits_mf.val),
            "mf_test_loss": self.dataset.loss(
                checkpoint["C_hat"], indices=replicate.splits_mf.test),
            "if_val_loss": self.dataset.loss(
                if_val, indices=replicate.splits_if.val, mode="if"),
            "if_test_loss": self.dataset.loss(
                if_test, indices=replicate.splits_if.test, mode="if")
        }
        return losses, checkpoint

    def train(self, key, replicate, tqdm=None):
        """Train model for a single swarm of k-CV replicates."""
        _train = vmap(self.epoch_train, in_axes=(0, 0, self.Replicate_vmap))
        _val = vmap(self.epoch_val, in_axes=(0, self.Replicate_vmap))

        k1, k2 = random.split(key, 2)
        state = vmap(self._init_)(split.keys(k1, self.k), replicate.if_train)

        iterator = random.split(k2, self.epochs)
        if tqdm is not None:
            iterator = tqdm(iterator)

        history = History(cpu=self.cpu)
        for ki in iterator:
            loss, state = _train(split.keys(ki, self.k), state, replicate)
            losses, checkpoint = _val(state, replicate)
            val = (
                losses["mf_val_loss"] * self.beta[0]
                + losses["if_val_loss"] * self.beta[1])
            history.log(train_loss=loss, **losses)
            history.update(val, **checkpoint)

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

        # Matrix Factorization Objective
        key, k1, k2 = random.split(key, 3)
        mf_train, mf_test = split.vmap_at_least_one(
            k1, dim=self.dataset.shape, replicates=self.replicates,
            train=int(self.dataset.size * p))

        # Have to do this step outside because fit synchronizes globally
        if do_baseline:
            _baseline = Rank1(self.dataset)
            m_bar, d_bar = _baseline.fit(mf_train)
            C_bar = _baseline.predict(m_bar, d_bar)
        else:
            m_bar = jnp.zeros(self.dataset.shape[0])
            d_bar = jnp.zeros(self.dataset.shape[1])
            C_bar = jnp.zeros_like(self.dataset)

        mf_train, mf_val = split.vmap_crossval(k1, mf_train, split=self.k)
        mf_splits = self.Splits(train=mf_train, val=mf_val, test=mf_test)

        # Interference Objective
        key, k1, k2 = random.split(key, 3)
        if_train, if_test = split.vmap_iid(
            k2, dim=self.dataset.if_size, replicates=self.replicates,
            train=int(self.dataset.if_size * p))
        if_train, if_val = split.vmap_crossval(k2, if_train, split=self.k)
        if_splits = self.Splits(train=if_train, val=if_val, test=if_test)

        # Actually train
        replicates = self.Replicate(
            m_bar=m_bar, d_bar=d_bar, splits_mf=mf_splits, splits_if=if_splits)
        results = vmap(partial(self.train, tqdm=tqdm))(
            split.keys(key, self.replicates), replicates)

        return {
            # Matrix Factorization Splits
            "mf_train": mf_train, "mf_val": mf_val, "mf_test": mf_test,
            # Interference Splits
            "if_train": if_train, "if_val": if_val, "if_test": if_test,
            "C_bar": C_bar, "m_bar": m_bar, "d_bar": d_bar,
            **results
        }
