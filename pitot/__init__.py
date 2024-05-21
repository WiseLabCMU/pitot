r"""Pitot: method and baselines.

    __________________      _____
    ___  __ \__(_)_  /________  /_
    __  /_/ /_  /_  __/  __ \  __/
    _  ____/_  / / /_ / /_/ / /_
    /_/     /_/  \__/ \____/\__/
.
"""

from . import presets
from .pitot import Pitot
from .ignore import PitotIgnore
from .monolith import Monolith
from .attention import Attention
from .linear_scaling import (
    LinearScaling, LinearScalingProblem, LinearScalingSolution)


__all__ = [
    "presets",
    "Pitot", "PitotIgnore", "Monolith", "Attention",
    "LinearScaling", "LinearScalingProblem", "LinearScalingSolution"
]

models = {
    "Pitot": Pitot,
    "PitotIgnore": PitotIgnore,
    "Monolith": Monolith,
    "Attention": Attention
}
