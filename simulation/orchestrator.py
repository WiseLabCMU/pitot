"""Orchestrator simulation."""

import numpy as np
from tqdm import tqdm
from jax import numpy as jnp
from jax import vmap

from beartype.typing import Optional
from jaxtyping import Float32, Array

from prediction import Objective
from simulation import Jobs, NetworkSimulation


class SimulatedOrchestrator:
    """Evalute simulated orchestrator.

    Parameters
    ----------
    objective: matrix factorization objective for error calculation.
    predictor: path to prediction checkpoint to use.
    key: value to fetch from checkpoint.
    p: calibration percentile.
    """

    def __init__(
        self, objective: Objective, network: NetworkSimulation,
        predictor: Optional[str] = None,
        key: str = "C_hat", p: int = 5, name: str = None
    ) -> None:
        self.network = network
        self.name = predictor if name is None else name

        if predictor is not None:
            npz = np.load(predictor)
            error = vmap(objective.error)(npz[key], jnp.array(npz["mf_test"]))
            margin = jnp.percentile(error, p, axis=1)
            pred = npz[key]
            if len(pred.shape) == 4:
                pred = np.mean(pred, axis=1)
            self.pred = network.dataset.to_ms(pred - margin.reshape(-1, 1, 1))
        else:
            self.pred = None

    def evaluate(
        self, jobs: list[Jobs]
    ) -> tuple[Float32[Array, "nb nj"], Float32[Array, "nb nd"]]:
        """Run evaluation on jobs."""
        res = []
        utilization = []
        for i, job in tqdm(enumerate(jobs), desc=self.name, total=len(jobs)):
            pred = None if self.pred is None else self.pred[i]
            assn = self.network.binpack(job, matrix=pred)
            t_pred = self.network.latency(job, device=assn)
            res.append(t_pred)

            module_util = self.network.utilization(job)[
                jnp.arange(assn.shape[0]), assn]
            utilization.append(
                self.network.capacity.at[assn].add(-module_util)
                / self.network.capacity)

        return jnp.stack(res), jnp.stack(utilization)
