"""Orchestrator simulation."""

import time

import numpy as np
from tqdm import tqdm
from jax import numpy as jnp
from jax import vmap

from beartype.typing import Optional
from jaxtyping import Float32, Array

from prediction import Objective
from .algorithms import (
    OrchestrationAlgorithm, CandidateAlgorithm,
    mask_none, greedy, unconstrained)
from .jobs import Jobs
from .system import NetworkSimulation


class SimulatedOrchestrator:
    """Evaluate simulated orchestrator.

    Parameters
    ----------
    objective: matrix factorization objective for error calculation.
    predictor: path to prediction checkpoint to use.
    log_predictor: whether predictor is in log form.
    key: value to fetch from checkpoint.
    p: calibration percentile.
    algorithm: orchestration algorithm to use.
    proposal: algorithm to propose valid mappings.
    name: display name.
    """

    def __init__(
        self, network: NetworkSimulation,
        objective: Optional[Objective] = None,
        predictor: Optional[str] = None, log_predictor: bool = True,
        key: str = "C_hat", p: int = 5,
        algorithm: OrchestrationAlgorithm = greedy,
        proposal: CandidateAlgorithm = mask_none,
        name: Optional[str] = None
    ) -> None:
        self.network = network
        self.name = predictor if name is None else name
        self.algorithm = algorithm
        self.proposal = proposal

        if predictor is not None:
            assert(objective is not None)
            npz = np.load(predictor)
            pred = (
                np.log(np.maximum(npz[key], 1e-3))
                if not log_predictor else npz[key])
            error = vmap(objective.error)(pred, jnp.array(npz["mf_test"]))
            margin = jnp.nanpercentile(error, p, axis=1)
            if len(pred.shape) == 4:
                pred = np.mean(pred, axis=1)
            self.pred = np.nan_to_num(np.array(
                network.dataset.to_ms(pred - margin.reshape(-1, 1, 1)) / 10
            )[:, self.network.bootstrap], nan=np.inf)
            # Remove invalid entries
            tmp = np.zeros((self.pred.shape[0], *self.network.valid.shape))
            tmp[:, ~self.network.valid] = np.inf
            self.pred += tmp
        else:
            self.pred = None

    def evaluate(
        self, jobs: list[Jobs], limit: float = 0.0
    ) -> tuple[Float32[Array, "nb nj"], Float32[Array, "nb nd"]]:
        """Run evaluation on jobs."""
        res, utilization, runtime = [], [], []
        for i, jobset in tqdm(
                enumerate(jobs), desc=self.name, total=len(jobs)):
            if self.pred is None:
                pred = self.network.matrix
            else:
                pred = self.pred[i % self.pred.shape[0]]

            start = time.perf_counter()
            assn = self.network.binpack(
                self.algorithm, jobset, matrix=pred, candidates=self.proposal)
            runtime.append(time.perf_counter() - start)

            if assn is None or np.any(assn == -1):
                res.append(np.ones(len(jobset.job)) * np.inf)
                utilization.append(np.zeros(len(self.network.network.tid)))
            else:
                t_pred = self.network.latency(jobset, device=assn)
                res.append(t_pred)

                total_util = np.sum(
                    np.nan_to_num(
                        self.network.utilization(jobset), posinf=0, neginf=0)
                    * np.identity(self.network.matrix.shape[0])[assn], axis=0)
                utilization.append(total_util / self.network.capacity)

        return np.stack(res), np.stack(utilization), runtime

    def summarize_results(
        self, latency: Float32[Array, "nb nj"],
        utilization: Float32[Array, "nb nd"], jobs: list[Jobs]
    ) -> dict:
        """Get summary statistics for latency traces."""
        latency_unconstrained = np.stack([
            self.network.latency(
                j, device=self.network.binpack(unconstrained, j))
            for j in jobs])
        rel = latency / latency_unconstrained

        return {
            "layer0": np.mean(utilization[:, self.network.network.layer == 0]),
            "layer1": np.mean(utilization[:, self.network.network.layer == 1]),
            "layer2": np.mean(utilization[:, self.network.network.layer == 2]),
            "mean": np.mean(rel, axis=1).tolist(),
            "median": np.median(rel, axis=1).tolist(),
            "percentile": np.percentile(rel.reshape(-1), [90, 95, 99]).tolist()
        }
