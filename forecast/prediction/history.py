"""Log training history."""

from jax import device_put
from jax import numpy as jnp


class History:
    """Training history.

    Parameters
    ----------
    cpu : jaxlib.xla_extension.Device or None
        Device to send data back to (to save GPU memory).
    """

    def __init__(self, cpu=None):

        self.history = {}
        self.cpu = cpu
        self.best = None

    def to_cpu(self, x):
        """Send object to CPU."""
        if self.cpu is not None:
            return device_put(x, self.cpu)
        else:
            return x

    def log(self, **kwargs):
        """Create new entry."""
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(device_put(v, device=self.cpu))

    def update(self, loss, **kwargs):
        """Overwrite if loss is improved."""
        _cpu = {k: device_put(v, device=self.cpu) for k, v in kwargs.items()}
        if self.best is None:
            self.history.update(_cpu)
        else:
            ind = loss < self.best
            for k, v in _cpu.items():
                self.history[k] = self.history[k] * (1 - ind) + v * ind

    def export(self):
        """Export values as nested structure with entries as axis 0."""
        return {k: jnp.array(v) for k, v in self.history.items()}
