"""Log training history."""

from jax import device_put
from jax import numpy as jnp

from beartype.typing import Optional
from jaxtyping import Float32, Array, Shaped, PyTree
from jaxlib.xla_extension import Device


class History:
    """Training history.

    Parameters
    ----------
    cpu: Device to send data back to (to save GPU memory).
    """

    def __init__(self, cpu: Optional[Device] = None) -> None:

        self.history: dict = {}
        self.cpu = cpu
        self.best = None

    def to_cpu(self, x: PyTree) -> PyTree:
        """Send object to CPU."""
        if self.cpu is not None:
            return device_put(x, self.cpu)
        else:
            return x

    def log(self, **kwargs) -> None:
        """Create new entry."""
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(device_put(v, device=self.cpu))

    def update(self, loss: Float32[Array, "batch"], **kwargs) -> None:
        """Overwrite if loss is improved."""
        _cpu = {k: device_put(v, device=self.cpu) for k, v in kwargs.items()}
        if self.best is None:
            self.history.update(_cpu)
        else:
            ind = loss < self.best
            for k, v in _cpu.items():
                self.history[k] = self.history[k] * (1 - ind) + v * ind

    def export(self) -> dict[str, Shaped[Array, "batch ..."]]:
        """Export values as nested structure with entries as axis 0."""
        return {k: jnp.array(v) for k, v in self.history.items()}
