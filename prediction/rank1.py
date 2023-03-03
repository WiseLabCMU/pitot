"""Rank 1 baseline model."""

from jax import numpy as jnp
from jax import vmap, jit

from beartype.typing import NamedTuple, Union, Optional

from jaxtyping import Float, Bool, Integer
from jaxtyping import Array


class Rank1Problem(NamedTuple):
    """Rank 1 baseline problem state.

    Attributes
    ----------
    mask: occupancy mask of which samples to use.
    n: number of samples in each row.
    m: number of samples in each column.
    """

    mask: Bool[Array, "nx ny"]
    n: Integer[Array, "nx"]
    m: Integer[Array, "ny"]


class Rank1Solution(NamedTuple):
    """Rank 1 baseline problem solution.

    Attributes
    ----------
    x: row features.
    y: column features.
    """

    x: Float[Array, "nx"]
    y: Float[Array, "ny"]

    def predict(
        self, indices: Optional[
            Union[Integer[Array, "2"], Integer[Array, "b 2"]]] = None
    ) -> Union[Float[Array, "b"], Float[Array, "nx ny"]]:
        """Generate prediction."""
        if indices is None:
            return self.x.reshape(-1, 1) + self.y.reshape(1, -1)
        elif indices.shape == (2,):
            return self.x[indices[0]] + self.y[indices[1]]
        else:
            return self.x[indices[:, 0]] + self.y[indices[:, 1]]


class Rank1:
    """Rank 1 baseline matrix factorization model Y_ij = x[i] + j[j].

    NOTE: Since jax uses 32-bit float by default, if the dataset magnitude is
    around ~1e-1 - 1e1, the best precision for the delta will be ~1e-7, so
    tol should be >>1e-7.

    Parameters
    ----------
    data : dataset matrix.
    init_val : Initial x and y values; should be set to 0.5 * E[A].
    max_iter : Max number of iterations.
    tol : Convergence criteria on l2 norm; stops when all replicates converge.
    """

    def __init__(
        self, data: Float[Array, "nx ny"], init_val: float = 0.,
        max_iter: int = 10, tol: float = 1e-5
    ) -> None:
        self.data = data
        self.init_val = init_val
        self.max_iter = max_iter
        self.tol = tol

    def init(
        self, mask: Bool[Array, "nx ny"]
    ) -> tuple[Rank1Problem, Rank1Solution]:
        """Create problem.

        Parameters
        ----------
        mask: mask indicating which entries are in the training set.

        Returns
        -------
        problem: created problem with pre-computed row/column counts.
        init: initial state.
        """
        problem = Rank1Problem(
            mask=mask,
            n=jnp.maximum(jnp.sum(mask, axis=1), 1),
            m=jnp.maximum(jnp.sum(mask, axis=0), 1))
        init = Rank1Solution(
            x=jnp.ones(self.data.shape[0]) * self.init_val,
            y=jnp.ones(self.data.shape[1]) * self.init_val)
        return problem, init

    def iter(
        self, problem: Rank1Problem, state: Rank1Solution
    ) -> tuple[Rank1Solution, Float[Array, ""]]:
        """Run single iteration."""
        x_new = jnp.sum(
            problem.mask * (self.data - state.y.reshape(1, -1)),
            axis=1) / problem.n
        y_new = jnp.sum(
            problem.mask * (self.data - x_new.reshape(-1, 1)),
            axis=0) / problem.m

        delta = (
            jnp.sum(jnp.abs(x_new - state.x))
            + jnp.sum(jnp.abs(y_new - state.y)))

        return Rank1Solution(x=x_new, y=y_new), delta

    def fit(self, mask: Bool[Array, "nx ny"]) -> Rank1Solution:
        """Fit in single-data mode."""
        problem, soln = self.init(mask)

        for _ in range(self.max_iter):
            soln, delta = self.iter(problem, soln)
            if delta < self.tol:
                return soln
        print(
            "Convergence warning: l1_delta={} after {} iterations; "
            "Increase tol or max_iter.".format(delta, self.max_iter))
        return soln

    def vfit(self, mask: Bool[Array, "batch nx ny"]) -> Rank1Solution:
        """Run vectorized fit.

        NOTE: This method MUST be performed at the highest level, i.e. cannot
        be vmapped, since it contains a break for the convergence criteria.
        """
        problem, soln = vmap(self.init)(mask)
        _iter = jit(vmap(self.iter))

        for _ in range(self.max_iter):
            soln, delta = _iter(problem, soln)
            if jnp.all(delta < self.tol):
                return soln
        print(
            "Convergence warning: max(l1_delta)={} after {} iterations; "
            "Increase tol or max_iter.".format(
                delta[delta > self.tol], self.max_iter))

        return soln
