"""Matrix completion model base class."""

from tqdm import tqdm as tqdm_base
from textwrap import indent

import jax
from jax import numpy as jnp
from jax import random
import optax

from beartype.typing import Optional, cast
from jaxtyping import PyTree, Float, Array, UInt

from . import types
from .loss import Loss
from .objective import ObjectiveSet, Split
from . import utils


class MatrixCompletionModel:
    """Matrix completion base class."""

    def __init__(
        self, objectives: ObjectiveSet, losses: Loss,
        optimizer: Optional[optax.GradientTransformation] = None,
        name: str = "MatrixCompletion"
    ) -> None:
        self.loss_func = losses
        self.objectives = objectives
        self.optimizer = (
            optax.adamaxw(0.001) if optimizer is None else optimizer)
        self.name = name

        self.step = jax.jit(self._step)
        self.val = jax.jit(self._val)

    def _init(
        self, key: random.PRNGKeyArray, splits: dict[str, Split]
    ) -> PyTree:
        """Get model parameters."""
        raise NotImplementedError()

    def init(
        self, key: random.PRNGKeyArray, splits: dict[str, Split]
    ) -> types.TrainState:
        """Get model initialization."""
        params = self._init(key, splits)
        opt_state = self.optimizer.init(params)
        return types.TrainState(params, opt_state)

    def evaluate(
        self, params: PyTree, data: dict[str, types.Data],
    ) -> dict[str, types.Predictions]:
        """Evaluate model predictions.

        Parameters
        ----------
        params: parameters for all embedding models.
        data: input indices (x) and actual values (y), organized by dataset.

        Returns
        -------
        Batch predictions for each loss.
        """
        raise NotImplementedError()

    def losses(
        self, params: PyTree, data: dict[str, types.Data],
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, "objs"]]]:
        """Compute objective losses and scalar loss for gradient descent."""
        predictions = self.evaluate(params, data)
        full = {
            k: jnp.mean(self.loss_func(y.y_true, y.y_hat), axis=0)
            for k, y in predictions.items()}
        scalar = self.objectives.loss({
            k: jnp.sum(v * self.loss_func.weight) for k, v in full.items()})
        return scalar, full

    def _step(
        self, key: random.PRNGKeyArray, state: types.TrainState,
        splits: dict[str, UInt[Array, "_"]]
    ) -> tuple[types.TrainState, dict]:
        """Run a single training step.

        NOTE: JAX doesn't like decorating class method; instead, setting
        ::
            self.step = jit(self._step)

        closes on ``self`` instead of passing on each call. Calling jit when
        used is technically correct, but leads to a ~50x penalty.
        """
        # Sample inside to make sure we catch it in the JIT!
        batch = self.objectives.sample(key, splits)

        # Close over data
        def _loss_func(params: PyTree) -> tuple[Float[Array, ""], dict]:
            scalar, full = self.losses(params, batch)
            return scalar, {"loss": scalar, **full}

        _loss_func(state.params)

        grads, aux = jax.grad(_loss_func, has_aux=True)(state.params)
        updates, opt_state = self.optimizer.update(
            grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return types.TrainState(params, opt_state), aux

    def _val(
        self, state: types.TrainState, splits: dict[str, UInt[Array, "_"]]
    ) -> tuple[Float[Array, ""], dict]:
        """Run a single validation step."""
        batch = self.objectives.index(splits)
        scalar, full = self.losses(state.params, batch)
        return scalar, {"loss": scalar, **full}

    def epoch_train(
        self, key: random.PRNGKeyArray, state: types.TrainState,
        splits: dict[str, UInt[Array, "_"]], steps: int = 100
    ) -> tuple[types.TrainState, dict]:
        """Train for a single (accounting) epoch."""
        acc = None
        for _ in range(steps):
            k, key = random.split(key, 2)
            state, aux = self.step(k, state, splits)
            if acc is None:
                acc = jax.tree_map(lambda x: x / steps, aux)
            else:
                acc = utils.tree_accumulate(acc, aux, divide=steps)

        return state, cast(dict, acc)

    def train(
        self, key: random.PRNGKeyArray, splits: dict[str, Split],
        state: Optional[types.TrainState] = None,
        steps: int = 10000, val_every: int = 100, tqdm=tqdm_base
    ) -> tuple[types.TrainState, PyTree]:
        """Run main training loop.

        Parameters
        ----------
        key: RNG seed.
        splits: train/val/test splits.
        state: initial parameters/optimizer state. If None, uses new init.
        state: Starting model/optimizer state. Creates a new state if None.
        steps: Number of steps to train for.
        val_every: Validate/report statistics every `val_every` steps.
        tqdm: Progress bar constructor.

        Returns
        -------
        state: post-optimization params/optimizer state (with the best val).
        log: train/val loss log.
        """
        if state is None:
            key, _k = random.split(key)
            state = self.init(_k, splits)

        train = {k: v.train for k, v in splits.items()}
        val = {k: v.val for k, v in splits.items()}

        log = []
        val_best = jnp.inf
        checkpoint_best = state
        for k in tqdm(random.split(key, steps // val_every)):
            state, aux = self.epoch_train(k, state, train, steps=val_every)
            val_loss, metrics = self.val(state, val)
            log.append({"train": aux, "val": metrics})

            if val_loss < val_best:
                checkpoint_best = state
                val_best = val_loss

        return checkpoint_best, utils.tree_stack(log)

    @classmethod
    def from_config(
        cls, objectives: ObjectiveSet, **kwargs
    ) -> "MatrixCompletionModel":
        """Create from configuration."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Get descriptive name."""
        return "MatrixCompletion::{}(\n{},\n{})".format(
            self.name, indent("objectives=" + str(self.objectives), ' ' * 4),
            indent("loss=" + str(self.loss_func), ' ' * 4))
