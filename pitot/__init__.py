"""Pitot methods."""

from . import presets
from .pitot import Pitot
from .ignore import PitotIgnore
from .monolith import Monolith
from .attention import Attention
from .linear_scaling import (
    LinearScaling, LinearScalingProblem, LinearScalingSolution)


__all__ = [
    "presets",
    "Pitot", "PitotIgnore",
    "LinearScaling", "LinearScalingProblem", "LinearScalingSolution"
]

models = {
    "Pitot": Pitot,
    "PitotIgnore": PitotIgnore,
    "Monolith": Monolith,
    "Attention": Attention
}
