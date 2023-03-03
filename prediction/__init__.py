"""Prediction library."""

from jaxtyping import install_import_hook

# Haiku (0.0.9) modules currently don't work with jaxtyped (0.2.8)
from prediction import models

with install_import_hook("prediction", ("beartype", "beartype")):
    from prediction.utils import apply_recursive, Index, Matrix
    from prediction.train import CrossValidationTrainer
    from prediction.dataset import Dataset


__all__ = [
    "apply_recursive", "Index", "Matrix",
    "models", "CrossValidationTrainer", "Dataset"
]
