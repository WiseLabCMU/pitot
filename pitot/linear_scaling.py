"""Rank 1 baseline model."""

import jax
import numpy as np
from beartype.typing import NamedTuple, Optional, Union
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, UInt

from prediction import types


class LinearScalingProblem(NamedTuple):
    """Rank 1 baseline problem state.

    Attributes
    ----------
    matrix: data matrix.
    mask: occupancy mask of which samples to use.
    n: number of samples in each row.
    m: number of samples in each column.
    """

    matrix: Float[Array, "nx ny"]
    mask: Bool[Array, "nx ny"]
    n: UInt[Array, "nx"]
    m: UInt[Array, "ny"]


class LinearScalingSolution(NamedTuple):
    """Rank 1 baseline problem solution.

    Attributes
    ----------
    x: row features.
    y: column features.
    """

    x: Float[Array, "nx"]
    y: Float[Array, "ny"]


class LinearScaling:
    """Linear scaling baseline matrix factorization model Y_ij = x[i] + j[j].

    NOTE: Since jax uses 32-bit float by default, if the dataset magnitude is
    around ~1e-1 - 1e1, the best precision for the delta will be ~1e-7, so
    tol should be >>1e-7.

    Parameters
    ----------
    shape: matrix shape.
    init_val : Initial x and y values; should be set to 0.5 * E[A].
    max_iter : Max number of iterations.
    tol : Convergence criteria on l2 norm; stops when all replicates converge.
    """

    def __init__(
        self, shape: tuple[int, int], init_val: float = 0.,
        max_iter: int = 1000, tol: float = 1e-5
    ) -> None:
        self.shape = shape
        self.init_val = init_val
        self.max_iter = max_iter
        self.tol = tol

    def init(
        self, data: types.Data
    ) -> tuple[LinearScalingProblem, LinearScalingSolution]:
        """Create problem.

        Parameters
        ----------
        mask: mask indicating which entries are in the training set.

        Returns
        -------
        problem: created problem with pre-computed row/column counts.
        init: initial state.
        """
        matrix = np.full(self.shape, -np.inf, dtype=np.float32)
        matrix[data.x["platform"], data.x["workload"]] = data.y
        mask = ~np.isinf(matrix)

        problem = LinearScalingProblem(
            matrix=jnp.array(matrix), mask=jnp.array(mask),
            n=jnp.maximum(jnp.sum(mask, axis=1), 1),
            m=jnp.maximum(jnp.sum(mask, axis=0), 1))
        init = LinearScalingSolution(
            x=jnp.full(matrix.shape[0], self.init_val),
            y=jnp.full(matrix.shape[1], self.init_val))
        return problem, init

    @staticmethod
    def iter(
        problem: LinearScalingProblem, state: LinearScalingSolution
    ) -> tuple[LinearScalingSolution, Float[Array, ""]]:
        """Run single iteration."""
        x_new = jnp.sum(
            problem.mask * (problem.matrix - state.y.reshape(1, -1)),
            axis=1) / problem.n
        y_new = jnp.sum(
            problem.mask * (problem.matrix - x_new.reshape(-1, 1)),
            axis=0) / problem.m

        delta = (
            jnp.sum(jnp.abs(x_new - state.x))
            + jnp.sum(jnp.abs(y_new - state.y)))

        return LinearScalingSolution(x=x_new, y=y_new), delta

    def fit(self, data: types.Data) -> LinearScalingSolution:
        """Fit in single-data mode."""
        problem, soln = self.init(data)

        delta = self.tol
        for _ in range(self.max_iter):
            soln, delta = jax.jit(self.iter)(problem, soln)
            if delta < self.tol:
                return soln
        print(
            "Convergence warning: l1_delta={} after {} iterations; "
            "Increase tol or max_iter.".format(delta, self.max_iter))
        return soln

    @staticmethod
    def evaluate(
        soln: Optional[LinearScalingSolution], xy: types.Data,
    ) -> Union[float, Float[Array, "N"]]:
        """Generate prediction.

        Parameters
        ----------
        soln: solution to evaluate. If None, return 0.
        xy: input indices (x) and actual values (y).
        """
        if soln is None:
            return 0.
        else:
            return (
                soln.x[xy.x["platform"]] + soln.y[xy.x["workload"]])[:, None]
