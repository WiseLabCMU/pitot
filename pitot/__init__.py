"""Pitot methods."""

from . import presets
from .pitot import Pitot
from .linear_scaling import (
    LinearScaling, LinearScalingProblem, LinearScalingSolution)

__all__ = [
    "presets",
    "Pitot",
    "LinearScaling", "LinearScalingProblem", "LinearScalingSolution"
]

models = {
    "Pitot": Pitot
}
