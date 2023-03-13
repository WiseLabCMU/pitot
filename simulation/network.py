"""Network simulation."""

import pandas as pd
import numpy as np
from jax import numpy as jnp
from jax import random, vmap

from beartype.typing import NamedTuple
from jaxtyping import Float32, Array, UInt32, Int16, UInt8, Bool


def _load_latency(latency_trace: str):
    df = pd.read_csv(latency_trace)
    dist = df["UE_Server_distance_km"]
    short = df[dist < 20]["latency_ms"]
    long = df[(dist >= 20) & (dist < 500)]["latency_ms"]
    longest = df[dist > 1000]["latency_ms"]
    return jnp.array(short), jnp.array(long), jnp.array(longest)


class NetworkTopology(NamedTuple):
    """Simulated Cellular Network with a 2-level tree Topology.

    Attributes
    ----------
    layer: layer assignments (0=cloud, 1=tower, 2=edge).
    latency: latency of each device (in milliseconds). For layer 1 devices,
        this is the RTT to the network root; for layer 2 devices, this is the
        RTT to the layer 1 "tower".
    t_latency: latency of each tower (indexed by tower id).
    tid: assignments of devices to towers; towers are assigned to themselves.
    """

    layer: UInt8[Array, "n"]
    latency: Float32[Array, "n"]
    t_latency: Float32[Array, "n1"]
    tid: Int16[Array, "n"]

    @classmethod
    def simulate(
        cls, key: UInt32[Array, "2"], speed: Float32[Array, "n"],
        stddev: float = 0.5, n0: int = 10, n1: int = 50, alpha: float = 1.0,
        latency_trace: str = "data/dataset_speedtest_ookla.csv"
    ) -> "NetworkTopology":
        """Create simulated network.

        Parameters
        ----------
        key: PRNGKey.
        speed: speed of devices; faster devices are assigned to layer 1.
        stddev: stddev of noise to add to `speed` before ordering.
        n0: Number of layer 0 devices
        n1: Number of layer 1 devices (all remaining are layer 2)
        alpha: Dirichlet concentration parameter.
        """
        k1, key = random.split(key, 2)
        speed_noised = speed + random.normal(k1, shape=speed.shape) * stddev
        order = jnp.argsort(speed_noised).astype(jnp.int16)
        layer0 = order[:n0]
        layer1 = order[n0:n1 + n0]
        layer2 = order[n1 + n0:]

        layer = (
            jnp.zeros(speed.shape, dtype=jnp.uint8)
            .at[layer1].set(1)
            .at[layer2].set(2))

        k1, k2, key = random.split(key, 3)
        weights = random.dirichlet(k1, jnp.ones(n1) * alpha)
        assignments = random.choice(
            k2, n1, p=weights, shape=layer2.shape).astype(jnp.int16)
        tid = (
            jnp.zeros(speed.shape, dtype=jnp.int16)
            .at[layer1].set(jnp.arange(n1, dtype=jnp.int16) + 1)
            .at[layer2].set(assignments + 1))

        k1, k2, k3 = random.split(key, 3)
        short, long, longest = _load_latency(latency_trace)
        latency = (
            jnp.zeros(speed.shape)
            .at[layer0].set(random.choice(k1, longest, shape=layer0.shape))
            .at[layer1].set(random.choice(k2, long, shape=layer1.shape))
            .at[layer2].set(random.choice(k3, short, shape=layer2.shape)))

        return cls(
            layer=layer, latency=latency, t_latency=latency[layer1], tid=tid)

    def rtt_matrix(self) -> Float32[Array, "n n"]:
        """Compute RTT matrix."""
        # Half round trip: latency to first node shared by i, j
        def _rtth(i, j):
            layer2 = (self.layer[i] == 2)
            same = (self.tid[i] != self.tid[j])
            return (i != j) * (
                self.latency[i] * (layer2 | same)
                + self.t_latency[i] * (layer2 & same))

        N = self.tid.shape[0]
        RTTh = vmap(vmap(_rtth))(*jnp.meshgrid(jnp.arange(N), jnp.arange(N)))
        return RTTh + RTTh.T

    def adjacency(self) -> Bool[Array, "n n"]:
        """Tower "adjacency" matrix."""
        return (self.tid.reshape(-1, 1) == self.tid.reshape(1, -1))

    def get_order(self):
        """Get order for displaying by cluster."""
        order = []
        for c in range(np.max(self.tid) + 1):
            order += np.where(self.tid == c)[0].tolist()
        return np.array(order)
