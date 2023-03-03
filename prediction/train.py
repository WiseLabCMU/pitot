"""Matrix Factorization Training."""

from tqdm import tqdm
from functools import partial

import jax
from jax import random, value_and_grad, vmap
from jax import numpy as jnp
import optax

from beartype.typing import NamedTuple, Optional, Callable, Union
from jaxtyping import Float32, Array, Integer, PyTree, UInt32
from jaxlib.xla_extension import Device
import haiku as hk

from .dataset import Dataset
from .rank1 import Rank1Solution, Rank1
from . import split
from .history import History


VmapSpec = Optional[int]


class TrainState(NamedTuple):
    """Training state."""

    params: PyTree
    opt_state: PyTree


class Replicate(NamedTuple):
    """Single training replicate.

    Attributes
    ----------
    baseline: rank 1 baseline for fitting residuals (if present).
    train: train set (indices).
    val: val set (indices).
    """

    baseline: Union[VmapSpec, Optional[Rank1Solution]]
    train: Union[VmapSpec, Integer[Array, "k ntrain 2"]]
    val: Union[VmapSpec, Integer[Array, "k nval 2"]]


class CrossValidationTrainer:
    """Training class for vectorizable k-fold cross validation.

    Parameters
    ----------
    dataset : Source dataset, shared between replicates.
    model : Callable that creates the model to use.
    optimizer: Optimizer to use.
    batch: Batch size
    replicates: Number of replicates to train.
    k: Number of folds for cross validation.
    do_baseline: Use baseline as starting point, and fit residuals only.
    cpu: CPU to use to save data. If None, uses first CPU (jax.devices('cpu')).
    """

    def __init__(
        self, dataset: Dataset, model: Callable[[], hk.Module],
        optimizer: Optional[optax.GradientTransformation] = None,
        batch: int = 256, replicates: int = 10, k: int = 10,
        do_baseline: bool = True, cpu: Optional[Device] = None
    ) -> None:

        def forward(*args, **kwargs):
            return model()(*args, **kwargs)

        self.model = hk.without_apply_rng(hk.transform(forward))
        self.dataset = dataset
        self.optimizer = optax.adam(0.001) if optimizer is None else optimizer

        self.batch = batch
        self.replicates = replicates
        self.k = k
        self.do_baseline = do_baseline
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

        self.step = jax.jit(self._step)
        self.epoch_val = jax.jit(self._epoch_val)

    def _init(self, key: UInt32[Array, "2"], replicate: Replicate):
        """Initialize model parameters and optimization state."""
        params = self.model.init(
            key, jnp.zeros((1, 5), dtype=int), baseline=replicate.baseline)
        opt_state = self.optimizer.init(params)
        return TrainState(params, opt_state)

    def _step(
        self, key: UInt32[Array, "2"], state: TrainState, repl: Replicate
    ) -> tuple[Float32[Array, ""], TrainState]:
        """Single training step.

        NOTE: JAX doesn't like decorating class method; instead, setting
        ::
            self.step = jit(self._step)

        closes on ``self`` instead of passing on each call. Calling jit when
        used is technically correct, but leads to a ~50x penalty.
        """
        k1, k2 = random.split(key, 2)
        ij = split.batch(k1, repl.train, batch=self.batch)

        # Close over all but params so they aren't included in value_and_grad.
        def _loss_func(params):
            pred = self.model.apply(params, ij, baseline=repl.baseline)
            return self.dataset.loss(pred, indices=ij)

        loss, grads = value_and_grad(_loss_func)(state.params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, TrainState(params, opt_state)

    def epoch_train(
        self, key: UInt32[Array, "2"], state: TrainState, repl: Replicate,
        epoch_size: int = 100
    ) -> tuple[Float32[Array, ""], TrainState]:
        """Single epoch (training only) for a single replicate."""
        epoch_loss = 0.
        for ki in random.split(key, epoch_size):
            loss, state = self.step(ki, state, repl)
            epoch_loss += loss
        return epoch_loss / epoch_size, state

    def _epoch_val(
        self, state: TrainState, repl: Replicate
    ) -> tuple[dict[str, Float32[Array, "..."]], dict]:
        """Post-epoch validation."""
        val, checkpoint = self.model.apply(
            state.params, repl.val, full=True, baseline=repl.baseline)
        losses = {"val_loss": self.dataset.loss(val, repl.val)}
        return losses, checkpoint

    def train(
        self, key: UInt32[Array, "2"], repl: Replicate, epochs: int = 100,
        epoch_size: int = 100
    ) -> dict:
        """Run training for k-fold CV set."""
        replicate_spec = Replicate(baseline=None, train=0, val=0)
        vepoch_train = vmap(
            partial(self.epoch_train, epoch_size=epoch_size),
            in_axes=(0, 0, replicate_spec))
        vepoch_val = jax.jit(
            vmap(self.epoch_val, in_axes=(0, replicate_spec)))

        k1, k2 = random.split(key, 2)
        state = vmap(self._init, in_axes=(0, replicate_spec))(
            split.keys(k1, self.k), repl)

        history = History(cpu=self.cpu)
        for ki in tqdm(random.split(k2, epochs)):
            loss, state = vepoch_train(split.keys(ki, self.k), state, repl)
            losses, checkpoint = vepoch_val(state, repl)
            val = losses["val_loss"]
            history.log(train_loss=loss, **losses)
            history.update(val, **checkpoint)

        return history.export()

    def train_replicates(
        self, epoch_size: int = 100, epochs: int = 100,
        key: Union[int, UInt32[Array, "2"]] = 42, p: float = 0.25
    ) -> dict:
        """Train replicates.

        Parameters
        ----------
        key: Root random key; if int, creates one.
        epochs: Number of epochs (not a "true" epoch; just for accounting).
            A checkpoint is saved (to main memory) after each epoch.
        epoch_size: Number of batches per epoch; each batch is IID.
        p: Target sparsity level (proportion of train+val set).

        Returns
        -------
        Results dictionary. Each value has replicates as the first axis,
        and k-fold cross validation replicates as the second axis if
        present for that key.
        """
        if isinstance(key, int):
            key = random.PRNGKey(key)

        k1, k2, k3 = random.split(key, 3)
        train, test = vmap(partial(
            split.split, data=self.dataset.indices,
            split=int(self.dataset.data.size * (1 - p))
        ))(key=split.keys(k1, self.replicates))

        if self.do_baseline:
            # max_iter=1000 is vastly overkill
            # ... but it's cheap so why not
            baseline = Rank1(self.dataset.data, max_iter=1000).vfit(
                vmap(self.dataset.to_mask)(train))
        else:
            baseline = None

        train, val = vmap(partial(split.crossval, k=self.k))(
            key=split.keys(k2, self.replicates), data=train)

        replicates = Replicate(baseline=baseline, train=train, val=val)
        result = vmap(
            partial(self.train, epochs=epochs, epoch_size=epoch_size)
        )(split.keys(k3, self.replicates), replicates)

        result.update({"train": train, "val": val, "test": test})
        if baseline:
            result.update({
                "C_bar": vmap(Rank1.predict)(baseline),
                "p_bar": baseline.x, "m_bar": baseline.y
            })
        return result
