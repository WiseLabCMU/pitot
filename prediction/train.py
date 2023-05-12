"""Matrix Factorization Training."""

from tqdm import tqdm as tqdm_base
from functools import partial

import jax
from jax import random, value_and_grad, vmap
from jax import numpy as jnp
import optax

from beartype.typing import NamedTuple, Optional, Callable, Union, cast
from jaxtyping import Float32, Array, Integer, PyTree
from jaxlib.xla_extension import Device
import haiku as hk

from .dataset import Dataset
from .rank1 import Rank1Solution, Rank1
from . import split
from .history import History
from .objective import Objective


#: Vmap axes specifications
VmapSpec = Optional[int]


class TrainState(NamedTuple):
    """Training state."""

    params: PyTree
    opt_state: PyTree


class Replicate(NamedTuple):
    """Single training replicate.

    All indices are (platform, module, interferers...).

    Attributes
    ----------
    baseline: rank 1 baseline for fitting residuals (if present).
    train: train set.
    val: val set.
    test: test set.
    """

    baseline: Optional[Rank1Solution]
    train: list[Integer[Array, "n1"]]
    val: list[Integer[Array, "n2"]]
    test: list[Integer[Array, "n3"]]


class CrossValidationTrainer:
    """Training class for vectorizable k-fold cross validation.

    Parameters
    ----------
    dataset: Source dataset, shared between replicates.
    model: Callable that creates the model to use.
    objective: Model objective functions; the first one should be a matrix
        factorization objective.
    optimizer: Optimizer to use.
    replicates: Number of replicates to train.
    k: Number of folds for cross validation.
    do_baseline: Use baseline as starting point, and fit residuals only.
    cpu: CPU to use to save data. If None, uses first CPU (jax.devices('cpu')).
    """

    def __init__(
        self, dataset: Dataset, model: Callable[[], hk.Module],
        objectives: list[Objective] = [],
        optimizer: Optional[optax.GradientTransformation] = None,
        replicates: int = 10, k: int = 10,
        do_baseline: bool = True, cpu: Optional[Device] = None
    ) -> None:

        def forward(*args, **kwargs):
            return model()(*args, **kwargs)

        self.model = hk.without_apply_rng(hk.transform(forward))
        self.dataset = dataset
        self.objectives = objectives
        assert len(objectives) > 0
        assert objectives[0].name == "mf"
        self.optimizer = optax.adam(0.001) if optimizer is None else optimizer

        self.replicates = replicates
        self.k = k
        self.do_baseline = do_baseline
        self.cpu = jax.devices('cpu')[0] if cpu is None else cpu

        self.step = jax.jit(self._step)
        self.epoch_val = jax.jit(self._epoch_val)

    def _init(self, key: split.PRNGSeed, replicate: Replicate):
        """Initialize model parameters and optimization state."""
        params = self.model.init(
            key, jnp.zeros((1, 5), dtype=int), baseline=replicate.baseline)
        opt_state = self.optimizer.init(params)
        return TrainState(params, opt_state)

    def _step(
        self, key: split.PRNGSeed, state: TrainState, repl: Replicate
    ) -> tuple[Float32[Array, ""], TrainState]:
        """Single training step.

        NOTE: JAX doesn't like decorating class method; instead, setting
        ::
            self.step = jit(self._step)

        closes on ``self`` instead of passing on each call. Calling jit when
        used is technically correct, but leads to a ~50x penalty.
        """
        idx = [
            split.batch(key, t, batch=obj.batch_size)
            for obj, t in zip(self.objectives, repl.train)]
        x = [obj.x[i] for obj, i in zip(self.objectives, idx)]

        # Close over all but params so they aren't included in value_and_grad.
        def _loss_func(params):
            pred = self.model.apply(params, x, baseline=repl.baseline)
            return sum(
                obj.loss(p, i)
                for obj, p, i in zip(self.objectives, pred, idx))

        loss, grads = value_and_grad(_loss_func)(state.params)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, TrainState(params, opt_state)

    def epoch_train(
        self, key: split.PRNGSeed, state: TrainState, repl: Replicate,
        epoch_size: int = 100
    ) -> tuple[Float32[Array, ""], TrainState]:
        """Single epoch (training only) for a single replicate."""
        epoch_loss: Float32[Array, ""] = jnp.array(0.)
        for ki in random.split(key, epoch_size):
            loss, state = self.step(ki, state, repl)
            epoch_loss += loss
        return epoch_loss / epoch_size, state

    def _epoch_val(
        self, state: TrainState, repl: Replicate
    ) -> tuple[Float32[Array, "..."], dict]:
        """Post-epoch validation."""
        val_test = (
            [obj.x[v] for v, obj in zip(repl.val, self.objectives)],
            [obj.x[v] for v, obj in zip(repl.test, self.objectives)])
        (val, test), checkpoint = self.model.apply(
            state.params, val_test, full=True, baseline=repl.baseline)

        for obj, t in zip(self.objectives, test):
            if obj.save is not None:
                checkpoint[obj.save] = t

        val_loss = sum(
            obj.loss(pred, idx)
            for obj, pred, idx in zip(self.objectives, val, repl.val))
        return val_loss, checkpoint

    def train(
        self, key: split.PRNGSeed, repl: Replicate, epochs: int = 100,
        epoch_size: int = 100, tqdm=tqdm_base
    ) -> dict:
        """Run training for k-fold CV set."""
        replicate_spec = Replicate(
            baseline=None, train=0, val=0, test=None)  # type: ignore
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
            val_loss, checkpoint = vepoch_val(state, repl)
            history.log(train_loss=loss, val_loss=val_loss)
            history.update(val_loss, **checkpoint)

        return history.export()

    def train_replicates(
        self, epoch_size: int = 100, epochs: int = 100,
        key: Union[int, random.PRNGKeyArray] = 42, p: float = 0.25,
        tqdm=tqdm_base
    ) -> dict:
        """Train replicates.

        Parameters
        ----------
        key: Root random key; if int, creates one.
        epochs: Number of epochs (not a "true" epoch; just for accounting).
            A checkpoint is saved (to main memory) after each epoch.
        epoch_size: Number of batches per epoch; each batch is IID.
        p: Target sparsity level (proportion of train+val set).
        tqdm: Progress bar to use.

        Returns
        -------
        Results dictionary. Each value has replicates as the first axis,
        and k-fold cross validation replicates as the second axis if
        present for that key.
        """
        if isinstance(key, int):
            key = random.PRNGKey(key)

        key, *k1s = random.split(key, len(self.objectives) + 1)
        train, test = list(zip(*[
            vmap(partial(
                split.split, data=obj.indices, split=int(obj.size * (1 - p))
            ))(key=split.keys(k, self.replicates))
            for obj, k in zip(self.objectives, k1s)
        ]))

        if self.do_baseline:
            mask = vmap(self.dataset.to_mask)(self.objectives[0].x[train[0]])
            # max_iter=1000 is vastly overkill
            # ... but it's cheap so why not
            baseline = Rank1(self.dataset.data, max_iter=1000).vfit(mask)
        else:
            baseline = None

        key, *k2s = random.split(key, len(self.objectives) + 1)
        train, val = list(zip(*[
            vmap(partial(
                split.crossval, k=self.k
            ))(key=split.keys(k, self.replicates), data=t)
            for t, k in zip(train, k2s)
        ]))

        replicates = Replicate(
            baseline=baseline,
            train=cast(list[Integer[Array, "n1"]], train),
            val=cast(list[Integer[Array, "n2"]], val),
            test=cast(list[Integer[Array, "n3"]], test))
        result = vmap(partial(
            self.train, epochs=epochs, epoch_size=epoch_size, tqdm=tqdm)
        )(split.keys(key, self.replicates), replicates)

        for obj, _train, _val, _test in zip(self.objectives, train, val, test):
            result[obj.name + "_train"] = _train
            result[obj.name + "_val"] = _val
            result[obj.name + "_test"] = _test

        if baseline:
            result.update({
                "C_bar": vmap(Rank1.predict)(baseline),
                "p_bar": baseline.x, "m_bar": baseline.y
            })
        return result
