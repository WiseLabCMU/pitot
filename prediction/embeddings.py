"""Prediction embedding models."""

import jax
from jax import numpy as jnp
import haiku as hk

from jaxtyping import Float32, Array, Integer
from beartype.typing import Callable, Sequence, Optional


class BaseEmbedding(hk.Module):
    """Embedding model."""

    def __call__(
        self, i: Optional[Integer[Array, "batch"]], **unused
    ) -> Float32[Array, "batch dim"]:
        """Get embeddings.

        Parameters
        ----------
        i: input indices.
        unused: other arguments.

        Returns
        -------
        Generated embeddings. `dim` can be reshaped as needed, i.e. for
        multi-objective learning.
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, **kwargs) -> Callable[..., "BaseEmbedding"]:
        """Bind configuration parameters."""
        def _model(*a, **k) -> "BaseEmbedding":
            return cls(*a, **k, **kwargs)  # type: ignore

        return _model


class MFEmbedding(BaseEmbedding):
    """Matrix factorization-style embedding.

    Parameters
    ----------
    N: number of entries.
    dim: embedding dimension.
    name: model name.
    """

    def __init__(
        self, N: int = 0, dim: int = 128, name: str = "MF"
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.N = N

    def __call__(
        self, i: Optional[Integer[Array, "batch"]], **unused
    ) -> Float32[Array, "batch dim"]:
        """Get embeddings."""
        embeddings: Float32[Array, "N dim"] = hk.get_parameter(
            "learned_features", init=hk.initializers.RandomUniform(0.0, 1.0),
            shape=(self.N, self.dim))
        if i is None:
            return embeddings
        else:
            return embeddings[i]

    @classmethod
    def from_config(
        cls, features: Float32[Array, "N D"], name: str = "Factorization"
    ) -> Callable[..., "MFEmbedding"]:
        """Bind configuration parameters."""
        def _model(dim: int) -> "MFEmbedding":
            return cls(N=features.shape[0], dim=dim, name=name)  # type: ignore

        return _model


class HybridEmbedding(BaseEmbedding):
    """Learned embedding model.

    The "hybrid embedding" takes input features, but also learns additional
    features which are indexed and concatenated.

    Parameters
    ----------
    features: side information for each entry.
    learned_features: number of extra learned features.
    layers: number of MLP layers.
    dim: embedding dimension.
    activation: activation function.
    name: model name (for weights PyTree)
    """

    def __init__(
        self, features: Float32[Array, "N D"], learned_features: int = 0,
        layers: Sequence[int] = [], activation=jax.nn.gelu, dim: int = 128,
        name: str = "Hybrid"
    ) -> None:
        super().__init__(name=name)

        self.features = features
        self.learned_features = learned_features

        layers = sum([
            [hk.Linear(d, name="mlp_{}".format(i)), activation]  # type: ignore
            for i, d in enumerate(layers)
        ], []) + [hk.Linear(dim, name="mlp_out")]                # type: ignore
        self.mlp = hk.Sequential(layers, name="mlp")             # type: ignore

    def __call__(
        self, i: Optional[Integer[Array, "batch"]], **unused
    ) -> Float32[Array, "batch dim"]:
        """Get embeddings.

        Parameters
        ----------
        i: input indices.

        Returns
        -------
        Generated embeddings in batch-output-embedding order.
        """
        learned_features = hk.get_parameter(
            "learned_features", init=hk.initializers.RandomUniform(0.0, 1.0),
            shape=(self.features.shape[0], self.learned_features))

        if i is None:
            return self.mlp(jnp.concatenate(
                [self.features, learned_features], axis=1))
        else:
            Xm: Float32[Array, "batch dim_obj"] = jnp.concatenate(
                [self.features[i], learned_features[i]], axis=1)
            return self.mlp(Xm)

    @classmethod
    def from_config(
        cls, features: Float32[Array, "N D"], activation: str = "gelu",
        learned_features: int = 0, layers: Sequence[int] = [],
        name: str = "Hybrid"
    ) -> Callable[..., "HybridEmbedding"]:
        """Bind configuration parameters."""
        def _model(dim: int) -> "HybridEmbedding":
            return cls(  # type: ignore
                features, activation=getattr(jax.nn, activation), name=name,
                learned_features=learned_features, layers=layers, dim=dim)

        return _model
