"""Matrix completion generalized objective."""

from textwrap import indent

import numpy as np
from beartype.typing import Iterable, NamedTuple, Optional, Union
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray, UInt

from . import types


class Split(NamedTuple):
    """Single training replicate splits."""

    train: UInt[Array, "N_train"]
    val: UInt[Array, "N_val"]
    test: UInt[Array, "N_test"]

    def as_dict(self) -> dict[str, UInt[Array, "_"]]:
        """Convert to dictionary for savez."""
        return {k: getattr(self, k) for k in ["train", "val", "test"]}

    @classmethod
    def from_npz(
        cls, path: str, objectives: Iterable[str] = []
    ) -> dict[str, "Split"]:
        """Load saved splits from npz file."""
        npz = np.load(path)

        def _load(obj):
            args = {
                k: npz["{}_{}".format(obj, k)]
                for k in ["train", "val", "test"]}
            return cls(**args)

        return {k: _load(k) for k in objectives}

    def __repr__(self):
        """Print split sizes only."""
        total = self.train.shape[0] + self.val.shape[0] + self.test.shape[0]
        return (
            "Split(n_train={} [{:.2f}%], n_val={} [{:.2f}%], n_test={} "
            "[{:.2f}%])".format(
                self.train.shape[0], 100 * self.train.shape[0] / total,
                self.val.shape[0], 100 * self.val.shape[0] / total,
                self.test.shape[0], 100 * self.test.shape[0] / total))


class Objective(NamedTuple):
    """Generalized runtime matrix completion objective.

    NOTE: for all dictionaries, the keys correspond to each axis in the matrix.

    Attributes
    ----------
    indices: index of data points by axis.
    t: observed runtime.
    data: value of features.
    features: correspondence between axis names and feature names.
    weight: optimization weight of this objective.
    batch: objective batch size for training.
    name: friendly objective name.
    """

    indices: dict[str, UInt[Array, "N"]]
    data: dict[str, Float[Array, "_ _"]]
    t: Float[Array, "N"]
    features: dict[str, str]
    weight: float
    batch: int
    name: str

    @property
    def shape(self, order: Optional[list[str]] = None) -> tuple[int, ...]:
        """Get matrix shape."""
        if order is None:
            order = list(self.indices.keys())
        return tuple(self.data[k].shape[0] for k in order)

    @classmethod
    def from_npz(
        cls, *path: str, axes: dict[str, str] = {}, name: str = "Objective",
        weight: float = 1.0, batch: int = 2048, log: bool = True
    ) -> "Objective":
        """Create from dataset file.

        Parameters
        ----------
        path: path(s) to .npz file.
        axes: matrix completion axes; keys correspond to axis names
            (e.g. platform, workload) and values correspond to feature names
            (e.g. opcounts).
        """
        data = {}
        for p in path:
            data.update(dict(np.load(p)))

        return cls(
            indices={k: jnp.array(data["i_{}".format(k)]) for k in axes},
            data={
                k: jnp.array(data["d_{}".format(v)]) for k, v in axes.items()},
            t=jnp.log(data["t"]) if log else jnp.array(data["t"]),
            features=axes, weight=weight, batch=batch, name=name)

    def split(
        self, key: PRNGKeyArray,
        train: Union[float, int] = 8000, val: Union[float, int] = 2000
    ) -> Split:
        """Create splits.

        Parameters
        ----------
        key: random seed
        ntrain: number of training data points.
        nval: number of validation data points.

        Returns
        -------
        Train/val/test splits, with all remaining data assigned to test.
        """
        if isinstance(train, float):
            train = int(self.t.shape[0] * train)
        if isinstance(val, float):
            val = int(self.t.shape[0] * val)

        shuffled = random.permutation(key, self.t.shape[0])
        itrain = shuffled[:train]
        ival = shuffled[train:train + val]
        itest = shuffled[train + val:]
        return Split(train=itrain, val=ival, test=itest)

    def index(  # type: ignore
        self, i: UInt[Array, "batch"]
    ) -> types.Data:
        """Index into dataset.

        Note the dataset is only indexed one layer deep (i.e. not
        to the actual features, but to the platform/workload index).
        """
        return types.Data(
            x={k: v[i] for k, v in self.indices.items()}, y=self.t[i])

    def to_matrix(
        self, order: Optional[list[str]] = None
    ) -> Float[np.ndarray, "..."]:
        """Arrange data into a matrix/tensor form.

        Parameters
        ----------
        order: optional axis order. If None, uses the indices iteration order.

        Returns
        -------
        Matrix with each axis in this objective as a dimension.
        - Entries that are not observed as returned as `np.nan`.
        - Entries that are observed multiple times may return either.
        """
        if order is None:
            order = list(self.indices.keys())
        mat = np.full([self.data[k].shape[0] for k in order], np.nan)
        mat[tuple(order)] = self.t  # type: ignore
        return mat

    def __repr__(self) -> str:
        """Get descriptive name."""
        features = ", ".join(
            ["{}={}".format(k, v.shape) for k, v in self.data.items()])
        return "Objective({}, N={}, {}, weight={}, batch={})".format(
            self.name, self.t.shape[0], features, self.weight, self.batch)


class ObjectiveSet:
    """Collection of objectives."""

    def __init__(
        self, objectives: dict[str, Objective],
    ) -> None:
        self.objectives = objectives
        self.total_weight = sum(v.weight for v in objectives.values())

    def __getitem__(self, key: str) -> Objective:
        """Shortcut for `ObjectiveSet.objectives[key]`."""
        return self.objectives[key]

    def items(self):
        """Pass through iterator."""
        return self.objectives.items()

    def values(self):
        """Pass through iterator."""
        return self.objectives.values()

    def index(
        self, splits: dict[str, UInt[Array, "_"]]
    ) -> dict[str, types.Data]:
        """Index data splits (for val/test)."""
        return {k: self.objectives[k].index(v) for k, v in splits.items()}

    def sample(
        self, key: PRNGKeyArray, splits: dict[str, UInt[Array, "_"]]
    ) -> dict[str, types.Data]:
        """Sample splits."""
        kt = random.split(key, len(self.objectives))
        batch_indices = {
            k: random.choice(seed, splits[k], shape=(v.batch,))
            for seed, (k, v) in zip(kt, self.objectives.items())}
        return self.index(batch_indices)

    def loss(
        self, batch: dict[str, Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Compute loss value."""
        acc: Float[Array, ""] = jnp.array(0.0)
        for k, v in batch.items():
            acc += self.objectives[k].weight * v
        return acc / self.total_weight

    def split(
        self, key: PRNGKeyArray,
        splits: dict[str, dict[str, Union[float, int]]]
    ) -> dict[str, Split]:
        """Create splits from size dictionary.

        Parameters
        ----------
        key: random seed.
        splits: 2-layer dictionary; the first layer specifies the objective,
            and the second layer specifies train, val split sizes.

        Returns
        -------
        Splits, organized by objective.
        """
        if isinstance(key, int):
            key = random.PRNGKey(key)

        seeds = random.split(key, len(splits))
        return {
            k: self.objectives[k].split(key=s, **v)
            for s, (k, v) in zip(seeds, splits.items())
        }

    @classmethod
    def from_config(cls, objectives: dict[str, dict] = {}) -> "ObjectiveSet":
        """Create from dataset config."""
        def _inner(cfg):
            return Objective.from_npz(
                *cfg["path"], **{k: v for k, v in cfg.items() if k != "path"})

        return ObjectiveSet({k: _inner(v) for k, v in objectives.items()})

    def __repr__(self) -> str:
        """Get descriptive name."""
        return "ObjectiveSet(\n{})".format(indent(
            '\n'.join([str(obj) for obj in list(self.objectives.values())]),
            ' ' * 4))
