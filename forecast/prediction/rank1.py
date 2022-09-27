"""Rank 1 Baseline Model."""

from jax import vmap, jit
from jax import numpy as jnp
from collections import namedtuple


class Rank1:
    """Rank 1 baseline matrix factorization model Y_ij = x[i] + j[j].

    Parameters
    ----------
    dataset : Dataset
        Training dataset.
    init_val : float
        Initial x and y values; should be set to 0.5 * E[A].
    max_iter : int
        Max number of iterations.
    tol : float
        Convergence criteria on l2 norm; stops when all replicates converge.

    NOTE: Since jax uses 32-bit float by default, if the dataset magnitude is
    around ~1e-1 - 1e1, the best precision for the delta will be ~1e-7, so
    tol should be >>1e-7.
    """

    Problem = namedtuple("Rank1Problem", ["mask", "N", "M"])

    def __init__(self, dataset, init_val=0., max_iter=100, tol=1e-5):
        self.dataset = dataset
        self.init_val = init_val
        self.max_iter = max_iter
        self.tol = tol

    def init(self, train):
        """Initialize rank1 problem."""
        mask = self.dataset.to_mask(train)
        problem = self.Problem(
            mask=mask,                  # Occupancy mask
            N=jnp.sum(mask, axis=1),    # Number of samples in each row
            M=jnp.sum(mask, axis=0)     # Number of samples in each column
        )

        # Initial state
        x = jnp.ones(self.dataset.shape[0]) * self.init_val
        y = jnp.ones(self.dataset.shape[1]) * self.init_val

        return problem, x, y

    def iter(self, problem, x, y):
        """Single alternating minimization iteration."""
        x_new = jnp.sum(
            problem.mask * (self.dataset.matrix - y.reshape(1, -1)),
            axis=1) / problem.N
        y_new = jnp.sum(
            problem.mask * (self.dataset.matrix - x_new.reshape(-1, 1)),
            axis=0) / problem.M

        l2_delta = jnp.sqrt(
            jnp.sum(jnp.square(x_new - x)) + jnp.sum(jnp.square(y_new - y)))

        return l2_delta, x_new, y_new

    def predict(self, x, y):
        """Generate predictions."""
        def _predict(_x, _y):
            return _x.reshape(-1, 1) + _y.reshape(1, -1)
        return vmap(_predict)(x, y)

    def fit(self, train):
        """Fit on training split; returns parameters.

        NOTE: This method MUST be performed at the highest level, i.e. cannot
        be vmapped, since it contains a break for the convergence criteria.
        """
        problem, x, y = vmap(self.init)(train)
        _iter = jit(vmap(self.iter))

        for _ in range(self.max_iter):
            l2_delta, x, y = _iter(problem, x, y)
            if jnp.all(l2_delta < self.tol):
                break
        else:
            print(
                "Convergence warning: l2_delta={} after {} iterations; "
                "Increase tol or max_iter.".format(
                    l2_delta[l2_delta > self.tol], self.max_iter))
        return x, y
