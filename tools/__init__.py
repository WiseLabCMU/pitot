"""Command line tools."""

from . import matrix
from . import opcodes
from . import platforms
from . import merge
from . import experiments
from . import summary
from . import simulate
from . import plot_simulation


commands = {
    "matrix": matrix,
    "merge": merge,
    "opcodes": opcodes,
    "platforms": platforms,
    "experiments": experiments,
    "summary": summary,
    "simulate": simulate,
    "plot_simulation": plot_simulation,
}
