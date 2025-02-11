r"""Pitot: method and baselines.

::

    __________________      _____
    ___  __ \__(_)_  /________  /_
    __  /_/ /_  /_  __/  __ \  __/
    _  ____/_  / / /_ / /_/ / /_
    /_/     /_/  \__/ \____/\__/
    Bringing Runtime Prediction
    up to speed for Edge Systems
.
"""

from . import presets
from .attention import Attention
from .ignore import PitotIgnore
from .linear_scaling import LinearScaling, LinearScalingProblem, LinearScalingSolution
from .monolith import Monolith
from .pitot import Pitot

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
