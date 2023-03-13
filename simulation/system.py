"""Scheduling simulation."""

import numpy as np
from jax import numpy as jnp
from jax import random

from beartype.typing import Union, NamedTuple
from jaxtyping import Float32, Array, UInt32, Int16

from prediction import Dataset, Rank1
from .network import NetworkTopology


class Jobs(NamedTuple):
    """Compute jobs.

    Attributes
    ----------
    src: source (event) device index.
    dst: destination (action) device index.
    job: module index.
    freq: frequency associated with this job.
    """

    src: Int16[Array, "b"]
    dst: Int16[Array, "b"]
    job: Int16[Array, "b"]
    freq: Float32[Array, "b"]


class NetworkSimulation:
    """Simulated Network.

    Parameters
    ----------
    key: random seed for network generation.
    dataset: path to source dataset.
    latency_trace: path to latency data.
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
        self, key: Union[int, UInt32[Array, "2"]] = 42,
        dataset: str = "data/data.npz",
        latency_trace: str = "data/dataset_speedtest_ookla.csv",
        n0: int = 10, n1: int = 50, alpha: float = 1.0,
        p_job: tuple[float, float, float] = (0.33, 0.33, 0.34),
        job_scale: float = 0.02,
        capacity: tuple[float, float, float] = (100.0, 4.0, 1.0),
    ) -> None:

        if isinstance(key, int):
            key = random.PRNGKey(key)

        # Load data, baseline
        self.dataset = Dataset.from_npz(dataset)
        self.valid = self.dataset.to_mask(self.dataset.x)

        self.matrix = self.dataset.to_ms(
            self.dataset.data).at[~self.valid].set(jnp.inf)
        self.N, self.M = self.matrix.shape

        self.rank1 = Rank1(self.dataset.data, max_iter=1000).fit(self.valid)

        # Make network
        key, _k = random.split(key, 2)
        self.network = NetworkTopology.simulate(
            _k, speed=self.rank1.x, alpha=alpha, n0=n0, n1=n1,
            latency_trace=latency_trace)
        self.rtt = self.network.rtt_matrix()

        # Job distribution
        self.P = self.__job_probability(*p_job)
        self.job_scale = job_scale

        # Compute
        self.capacity = sum(
            c * (self.network.layer == i) for i, c in enumerate(capacity))

    def __job_probability(
        self, p_same: float = 0.33, p_tower: float = 0.33, p_any: float = 0.34
    ) -> Float32[Array, "n n"]:
        """Create P matrix."""
        adj = self.network.adjacency()
        return (
            p_any * jnp.ones(adj.shape) / adj.shape[1]
            + p_tower * adj / jnp.sum(adj, axis=1)
            + p_same * jnp.eye(adj.shape[0]))

    def utilization(self, jobs: Jobs, matrix=None) -> Float32[Array, "b n"]:
        """Compute (predicted) resource utilization on each device."""
        if matrix is None:
            matrix = self.matrix
        return matrix[:, jobs.job].T * jobs.freq.reshape(-1, 1)

    def latency(
        self, jobs: Jobs, matrix=None, device=None
    ) -> Float32[Array, "b n"]:
        """Compute (predicted) latency on each device."""
        if matrix is None:
            matrix = self.matrix
        matrix = matrix.at[~self.valid].set(jnp.inf)

        if device is None:
            return (
                matrix[:, jobs.job].T
                + self.rtt[jobs.src]
                + self.rtt[jobs.dst])
        else:
            return (
                matrix[device, jobs.job]
                + self.rtt[jobs.src, device]
                + self.rtt[device, jobs.dst])

    def create_jobs(
        self, key: UInt32[Array, "2"], jobs: int = 100, freq_std: float = 0.5
    ) -> Jobs:
        """Sample random jobs."""
        k1, k2, k3, k4 = random.split(key, 4)

        src = random.choice(k2, self.matrix.shape[0], shape=(jobs,))
        dst = random.categorical(k3, jnp.log(self.P[src]), axis=1)
        job = random.choice(k4, self.matrix.shape[1], shape=(jobs,))

        freq_noise = random.normal(k1, shape=(jobs,)) * freq_std
        job_freq = self.job_scale * jnp.exp(freq_noise - self.rank1.y[job])

        return Jobs(src=src, dst=dst, job=job, freq=job_freq)

    def binpack(
        self, jobs: Jobs, matrix=None
    ) -> tuple[Int16[Array, "n"], Float32[Array, "n"]]:
        """Perform greedy bin packing.

        Greedy assignment is ordered by the rank1 estimate.

        NOTE: this procedure has to be done iteratively, and is very slow.
        """
        latency = self.latency(jobs, matrix=matrix)
        util = self.utilization(jobs, matrix=matrix)

        capacity = np.array(self.capacity)
        assn = np.ones(len(jobs.job), dtype=np.int16) * -1
        order = np.argsort(jobs.freq * np.exp(self.rank1.y[jobs.job]))
        for i in order:
            options = np.argsort(latency[i])
            choice = np.argmax(capacity[options] > util[i][options])
            remaining = capacity[options[choice]] - util[i, options[choice]]
            if remaining > 0:
                capacity[options[choice]] = remaining
                assn[i] = options[choice]
        return assn
