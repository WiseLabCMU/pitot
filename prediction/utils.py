"""Miscellaneous utilities."""

from functools import partial
from jaxtyping import PyTree, Integer, Array, Num

import jax
import numpy as np
from jax import numpy as jnp


def tree_stack(trees: list[PyTree], _np=jnp) -> PyTree:
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    """
    treedef = jax.tree_util.tree_structure(trees[0])
    leaves = [jax.tree_util.tree_flatten(t)[0] for t in trees]

    result_leaves = list(map(_np.stack, zip(*leaves)))
    return jax.tree_util.tree_unflatten(treedef, result_leaves)


def tree_concatenate(trees: list[PyTree]) -> PyTree:
    """Takes a list of trees and concatenates every corresponding leaf."""
    treedef = jax.tree_util.tree_structure(trees[0])
    leaves = [jax.tree_util.tree_flatten(t)[0] for t in trees]

    result_leaves = list(map(partial(jnp.concatenate, axis=0), zip(*leaves)))
    return jax.tree_util.tree_unflatten(treedef, result_leaves)


@jax.jit
def tree_accumulate(acc: PyTree, add: PyTree, divide=1.0) -> PyTree:
    """Accumulate tree values (JIT friendly)."""
    return jax.tree_map(lambda x, y: x + y / divide, acc, add)


class RangeAllocator:
    """Helper class that keeps track of allocated indices."""

    def __init__(self):
        self.acc: int = 0

    def allocate(self, *n: int) -> list[Integer[Array, "_"]]:
        """Allocate range."""
        res: list[Integer[Array, "_"]] = []
        for ni in n:
            res.append(jnp.arange(ni) + self.acc)
            self.acc += ni
        return res


def dict_flatten(x: dict, prefix: list[str] = []) -> dict:
    """Flatten dictionary with string keys to '_'-separated 'paths'."""
    res = {}
    for k, v in x.items():
        if isinstance(v, dict):
            res.update(dict_flatten(v, prefix=prefix + [k]))
        else:
            res['_'.join(prefix + [k])] = v
    return res


def tree_size(x: PyTree) -> int:
    """Get size (# parameters) of PyTree."""
    return sum(v.size for v in jax.tree_util.tree_leaves(x))


def array_unpack(
    x: Num[Array, "batch dim"], shape: tuple[int, ...]
) -> tuple[Num[Array, "batch d1"], Num[Array, "batch ..."]]:
    """Index entries from a batched array with a given shape."""
    size = np.prod(shape)
    x1 = x[:, :size]
    x2 = x[:, size:]
    return x2, x1.reshape(x.shape[0], *shape)
