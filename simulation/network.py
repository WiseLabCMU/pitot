"""Network simulation."""

import pandas as pd
import numpy as np

from beartype.typing import NamedTuple
from jaxtyping import Float32, UInt32, Int16, UInt8, Bool, Integer
from numpy import ndarray as Array


def _load_latency(
    latency_trace: str
) -> tuple[Float32[Array, "n"], Float32[Array, "n"], Float32[Array, "n"]]:
    """Load latency trace."""
    df = pd.read_csv(latency_trace)
    dist = df["UE_Server_distance_km"]
    short = df[dist < 20]["latency_ms"]
    long = df[(dist >= 20) & (dist < 500)]["latency_ms"]
    longest = df[dist > 1000]["latency_ms"]
    return (
        np.array(short, dtype=np.float32),
        np.array(long, dtype=np.float32),
        np.array(longest, dtype=np.float32))


def _sample_latency(
    random: np.random.Generator, latency_trace: str, layers: tuple[
        Integer[Array, "n0"], Integer[Array, "n1"], Integer[Array, "n2"]]
) -> Float32[Array, "n"]:
    """Sample latency values from layer assignments."""
    l0, l1, l2 = layers

    short, long, longest = _load_latency(latency_trace)
    latency = np.zeros(len(l0) + len(l1) + len(l2), dtype=np.float32)
    latency[l0] = random.choice(longest, size=l0.shape)
    latency[l1] = random.choice(long, size=l1.shape)
    latency[l2] = random.choice(short, size=l2.shape)
    return latency


def _rtt_matrix(
    layer: UInt8[Array, "n"], tid: Int16[Array, "n"],
    latency: Float32[Array, "n"], twr_latency: Float32[Array, "n1"]
) -> Float32[Array, "n n"]:
    """Compute RTT Matrix."""
    # Half round trip: latency to first node shared by i, j
    is_layer2 = (layer == 2)
    is_sametower = (tid.reshape(-1, 1) == tid.reshape(1, -1))
    is_samedevice = np.identity(len(layer), dtype=bool)
    RTTh: Float32[Array, "n n"] = (
        # All devices need to go up one layer unless it's the same device.
        latency * (~is_samedevice)
        # Devices in layer 2 also need to move up one more layer the root
        # unless the target device is on the same tower.
        + twr_latency[tid - 1] * (~is_sametower & is_layer2))

    return RTTh + RTTh.T


class NetworkTopology(NamedTuple):
    """Simulated Cellular Network with a 2-level tree Topology.

    Attributes
    ----------
    layer: layer assignments (0=cloud, 1=tower, 2=edge).
    latency: latency of each device (in milliseconds). For layer 1 devices,
        this is the RTT to the network root; for layer 2 devices, this is the
        RTT to the layer 1 "tower".
    tid: assignments of devices to towers; towers are assigned to themselves.
        Tower 0 indicates all "cloud" devices that are directly connected
        to the logical root.
    rtt: RTT matrix.
    """

    layer: UInt8[Array, "n"]
    latency: Float32[Array, "n"]
    tid: Int16[Array, "n"]
    rtt: Float32[Array, "n n"]

    @classmethod
    def simulate(
        cls, random: np.random.Generator, speed: Float32[Array, "n"],
        stddev: float = 0.5, n0: int = 10, n1: int = 50, alpha: float = 0.5,
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
        speed_noised = speed + random.normal(size=speed.shape) * stddev
        order = np.argsort(speed_noised)
        layers = (order[:n0], order[n0:n1 + n0], order[n1 + n0:])

        layer = np.zeros(speed.shape, dtype=np.uint8)
        layer[layers[1]] = 1
        layer[layers[2]] = 2

        weights = random.dirichlet(np.ones(n1) * alpha)
        assignments = random.choice(n1, p=weights, size=layers[2].shape)

        tid = np.zeros(speed.shape, dtype=np.int16)
        tid[layers[1]] = np.arange(n1) + 1
        tid[layers[2]] = assignments + 1

        latency = _sample_latency(random, latency_trace, layers)
        rtt = _rtt_matrix(
            layer=layer, tid=tid, latency=latency,
            twr_latency=latency[layers[1]])

        return cls(layer=layer, latency=latency, tid=tid, rtt=rtt)

    def adjacency(self) -> Bool[Array, "n n"]:
        """Tower "adjacency" matrix."""
        return (self.tid.reshape(-1, 1) == self.tid.reshape(1, -1))

    def get_order(self) -> Integer[Array, "n"]:
        """Get order for displaying by cluster."""
        order = []
        for c in range(np.max(self.tid) + 1):
            order += np.where(self.tid == c)[0].tolist()
        return np.array(order)
