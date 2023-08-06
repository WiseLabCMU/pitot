"""Scheduling simulation."""

import numpy as np

from beartype.typing import Union, Optional
from jaxtyping import Float32, Int16, Bool
from numpy import ndarray as Array

from prediction import Dataset, Rank1

from .jobs import JobSpec, Jobs
from .network import NetworkTopology
from .algorithms import CandidateAlgorithm, OrchestrationAlgorithm, mask_none


def _job_probability(
    adj: Bool[Array, "Np Np"],
    p_same: float = 0.33, p_tower: float = 0.33, p_any: float = 0.34
) -> Float32[Array, "Np Np"]:
    """Create job destination given source probability matrix."""
    return (
        p_any * np.ones(adj.shape) / adj.shape[1]
        + p_tower * adj / np.sum(adj, axis=1)
        + p_same * np.eye(adj.shape[0]))


class NetworkSimulation:
    """Simulated Network.

    Parameters
    ----------
    key: random seed for network generation.
    dataset: path to source dataset.
    latency_trace: path to latency data.
    n: total number of devices; sampled with replacement (i.e. bootstrapped).
    n0: number of devices in layer 0 (cloud).
    n1: number of devices in layer 1 (5G edge).
    alpha: concentration parameter for tower devices.
    p_job: probability of jobs being in anywhere, in the same cluster,
        or on the same device.
    job_scale: capacity requirements of each job as a multiplier of the
        execution time.
    capacity: capacity of layer 0, layer 1, and layer 2 devices.
    """

    def __init__(
        self, random: Union[np.random.Generator, int] = 42,
        dataset: str = "data/data.npz",
        latency_trace: str = "data/dataset_speedtest_ookla.csv",
        n: int = 200, n0: int = 10, n1: int = 50, alpha: float = 1.0,
        p_job: tuple[float, float, float] = (0.33, 0.33, 0.34),
        job_scale: float = 0.02,
        capacity: tuple[float, float, float] = (1000, 4.0, 1.0),
    ) -> None:
        random = np.random.default_rng(random)

        # Load data, baseline
        self.dataset = Dataset.from_npz(dataset)
        valid = np.array(self.dataset.to_mask(self.dataset.x))

        mat = np.array(
            self.dataset.to_ms(self.dataset.data) / 10, dtype=np.float32)
        mat[~valid] = np.inf

        rank1 = Rank1(self.dataset.data, max_iter=1000).fit(valid)

        # Bootstrap devices to hit target n
        self.bootstrap = random.integers(mat.shape[0], size=n)
        self.matrix = mat[self.bootstrap]

        self.network = NetworkTopology.simulate(
            random, speed=np.array(rank1.x)[self.bootstrap], alpha=alpha,
            n0=n0, n1=n1, latency_trace=latency_trace)

        self.jobspec = JobSpec(
            weight=_job_probability(self.network.adjacency(), *p_job),
            freq_mean=-np.array(rank1.y) + np.log(job_scale), freq_std=0.5)

        self.capacity = np.zeros_like(self.network.layer, dtype=np.float32)
        for i, c in enumerate(capacity):
            self.capacity[self.network.layer == i] = c

        # Interpolate data on the cloud (L0) to make sure all jobs always have
        # somewhere to go
        self.valid = valid[self.bootstrap]
        self.matrix[self.network.layer == 0] = np.nan_to_num(
            self.matrix[self.network.layer == 0], posinf=0.0)
        interp_mask = ~self.valid[self.network.layer == 0]
        interp = Rank1.predict(rank1)[self.bootstrap][self.network.layer == 0]
        self.matrix[self.network.layer == 0] += interp * interp_mask
        self.valid[self.network.layer == 0] = True

    def utilization(
        self, jobs: Jobs, matrix: Optional[Float32[Array, "Nj Np"]] = None
    ) -> Float32[Array, "b n"]:
        """Compute (predicted) resource utilization on each device."""
        if matrix is None:
            matrix = self.matrix
        else:
            # Don't allow over-packing
            matrix = np.maximum(self.matrix, matrix)

        return matrix[:, jobs.job].T * jobs.freq.reshape(-1, 1)

    def latency(
        self, jobs: Jobs, matrix: Optional[Float32[Array, "Nj Np"]] = None,
        device: Optional[Int16[Array, "Nj"]] = None
    ) -> Union[Float32[Array, "b n"], Float32[Array, "b"]]:
        """Compute (predicted) latency on each device."""
        if matrix is None:
            matrix = self.matrix

        rtt = self.network.rtt
        if device is None:
            return matrix[:, jobs.job].T + rtt[jobs.src] + rtt[jobs.dst]
        else:
            return (
                matrix[device, jobs.job]
                + rtt[jobs.src, device] + rtt[device, jobs.dst])

    def binpack(
        self, algorithm: OrchestrationAlgorithm, jobs: Jobs,
        matrix: Optional[Float32[Array, "Nj Np"]] = None,
        candidates: CandidateAlgorithm = mask_none
    ) -> tuple[Int16[Array, "n"], Float32[Array, "n"]]:
        """Run bin packing."""
        latency = self.latency(jobs, matrix=matrix)
        util = self.utilization(jobs, matrix=matrix)
        if candidates is not None:
            mask = candidates(jobs, self.network.tid, self.network.layer)
        else:
            mask = None
        return algorithm(latency, util, np.copy(self.capacity), mask=mask)
