"""Official Jobs Program for the simulator."""

import numpy as np

from beartype.typing import NamedTuple
from jaxtyping import Float32, Int16
from numpy import ndarray as Array


class Jobs(NamedTuple):
    """Compute jobs.

    Attributes
    ----------
    src: source (event) device index.
    dst: destination (action) device index.
    job: module index.
    freq: frequency that each job is executed.
    """

    src: Int16[Array, "Nj"]
    dst: Int16[Array, "Nj"]
    job: Int16[Array, "Nj"]
    freq: Float32[Array, "Nj"]


class JobSpec(NamedTuple):
    """Job simulation parameters.

    Attributes
    ----------
    weight: entry (i, j) indicates P[dst = platform j | src = platform i]
    freq_mean: "mean" log-frequency to run each job
    freq_noise: log-noise for each job
    """

    weight: Float32[Array, "Np Np"]
    freq_mean: Float32[Array, "Nj"]
    freq_std: float

    def simulate(self, random: np.random.Generator, n: int = 100) -> Jobs:
        """Simulate job from job parameters.
        
        NOTE: we have to implement the gumbel-max trick ourselves since numpy
        doesn't seem to come with one, and entering JAX is too expensive.
        """
        src = random.choice(self.weight.shape[0], size=(n,))
        _logits = np.log(self.weight[src])
        _gumbels = random.gumbel(size=_logits.shape)
        dst = np.argmax(_logits + _gumbels, axis=1)
        job = random.choice(self.freq_mean.shape[0], size=(n,))

        freq_noise = random.normal(size=(n,)) * self.freq_std
        job_freq = np.exp(freq_noise + self.freq_mean[job])

        return Jobs(
            src=src.astype(np.int16), dst=dst.astype(np.int16),
            job=job.astype(np.int16), freq=job_freq.astype(np.float32))
