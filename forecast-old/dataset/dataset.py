"""Dataset for prediction."""

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt


class Dataset:

    def __init__(self):

        # Interference
        if if_data is not None:
            self.if_data = self._load(if_data)
            self.if_ijk = jnp.concatenate([
                jnp.stack([
                    self.if_data["module"],
                    self.if_data["runtime"]
                ]).T,
                self.if_data["interferer"]
            ], axis=1)
            self.interference = jnp.log(
                self._get_key(self.if_data, key)) - jnp.log(offset)
            self.if_size = self.interference.shape[0]

    def index_if(self, indices):
        """Index into interference data."""
        if self.if_data is None:
            return None
        else:
            return self.if_ijk[indices]

    def _index(self, pred, indices=None, mode="mf"):
        if mode == "if":
            if indices is not None:
                return pred, self.interference[indices]
            else:
                return pred, self.interference
