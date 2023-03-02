"""Miscellaneous utilities."""

import os
import numpy as np

from jaxtyping import Shaped
from beartype.typing import (
    Callable, Optional, Sequence, Union, NamedTuple, Any)
from beartype import beartype


@beartype
def apply_recursive(
    path: str, func: Callable[[str], Optional[Any]],
    exclude: set[str] = {"runtimes.json"}
) -> list:
    """Apply function recursively in file system."""
    res = []
    for p in os.listdir(path):
        pp = os.path.join(path, p)
        if os.path.isdir(pp):
            res += apply_recursive(pp, func, exclude=exclude)
        else:
            if p not in exclude:
                d = func(pp)
                if d is not None:
                    res.append(d)
    return res


@beartype
class Index:
    """Enumerated value array indexing.

    Parameters
    ----------
    items: indexing item names.
    display: display names for items.
    """

    def __init__(
        self, items: Union[Sequence, np.ndarray],
        display: Optional[Union[Sequence, np.ndarray]] = None,
        name: str = "index"
    ) -> None:
        self.key = np.array(items)
        self.display = np.array(
            items if display is None else display)
        self.index = {n: i for i, n in enumerate(self.key)}

    @classmethod
    def from_objects(
            cls, objs: list[dict], key: str,
            display: Optional[Callable[[str], str]] = None):
        """Create from list of objects.

        Parameters
        ----------
        objs: objects (dictionaries) to create from.
        key: attribute to fetch from each object.
        display: function that transforms items to their final display names.
        """
        items = sorted(list(set([obj[key] for obj in objs])))
        if display is None:
            display = items
        else:
            display = [display(i) for i in items]
        return cls(items, display=display)

    def __matmul__(self, B):
        """Set product is denoted by A @ B."""
        items = [".".join([a, b]) for a in self.key for b in B.key]
        display = [
            "{}, {}".format(a, b) for a in self.display for b in B.display]
        return Index(items, display=display)

    def __len__(self) -> int:
        """Length indicates the set size."""
        return len(self.key)

    def __getitem__(self, key: Union[np.ndarray, slice, int, str]):
        """Index is indexable by item index or value."""
        if isinstance(key, np.ndarray) or isinstance(key, slice):
            return Index(self.key[key], display=self.display[key])
        elif isinstance(key, int):
            return self.key[key]
        else:
            return self.index[key]

    def set_xticks(self, ax):
        """Set this enum as xticks."""
        ax.set_xticks(np.arange(len(self.key)))
        ax.set_xticklabels(self.display, rotation="vertical")

    def set_yticks(self, ax):
        """Set this enum as yticks."""
        ax.set_yticks(np.arange(len(self.key)))
        ax.set_yticklabels(self.display)


MatrixSlice = Union[slice, np.ndarray]


@beartype
class Matrix(NamedTuple):
    """Matrix with enumerated labels.

    Attributes
    ----------
    data: matrix
    rows: row labels
    cols: column labels
    """

    data: Shaped[np.ndarray, "n m"]
    rows: Index
    cols: Index

    def plot(self, ax, xlabel=True, ylabel=True):
        """Draw plot."""
        ax.imshow(self.data)
        if ylabel:
            self.rows.set_yticks(ax)
        if xlabel:
            self.cols.set_xticks(ax)

    def __getitem__(
        self, val: Union[MatrixSlice, tuple[MatrixSlice, MatrixSlice]]
    ) -> "Matrix":
        """Index cylinder set intersections with numpy arrays (or slices)."""
        if isinstance(val, MatrixSlice):
            val = (val, slice(None, None, None))
        rows, cols = val

        return Matrix(
            data=self.data[rows][:, cols],
            rows=self.rows[rows], cols=self.cols[cols])

    def __matmul__(
        self, transform: Callable[
            [Shaped[np.ndarray, "n m"]], Shaped[np.ndarray, "n m"]]
    ) -> "Matrix":
        """Matrix multiplication operator denotes function composition."""
        return Matrix(
            data=transform(self.data), rows=self.rows, cols=self.cols)

    def save(
        self, path: str, data: str = "data",
        rows: str = "rows", cols: str = "cols"
    ) -> None:
        """Save matrix, naming keys according to parameters given."""
        np.savez(path, **{data: self.data, rows: self.rows, cols: self.cols})

    @classmethod
    def from_npz(
        cls, path: str, data: str = "data",
        rows: str = "rows", cols: str = "cols"
    ) -> "Matrix":
        """Load from npz file with specified keys."""
        npz = np.load(path)
        return cls(
            data=npz[data], rows=Index(npz[rows]), cols=Index(npz[cols]))
