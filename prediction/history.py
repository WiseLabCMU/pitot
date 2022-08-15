"""Log training history."""

from jax import device_put
from jax import numpy as jnp


class History:
    """Training history.

    Parameters
    ----------
    keys : str[]
        List of keys to log.
    cpu : jaxlib.xla_extension.Device or None
        Device to send data back to (to save GPU memory).
    """

    def __init__(self, keys, cpu=None):

        self.history = {k: [] for k in keys}
        self.cpu = cpu

    def to_cpu(self, x):
        """Send object to CPU."""
        if self.cpu is not None:
            return device_put(x, self.cpu)
        else:
            return x

    def log(self, **kwargs):
        """Create new entry."""
        for k, v in kwargs.items():
            self.history[k].append(device_put(v, device=self.cpu))

    def export(self):
        """Export values as nested structure with entries as axis 0."""
        return {k: jnp.array(v) for k, v in self.history.items()}
