"""Command line tools."""

from . import matrix
from . import opcodes
from . import platforms
from . import merge
from . import plot_dataset
from . import experiments
from . import summary
from . import compare
from . import figures
from . import tsne
from . import simulate
from . import plot_simulation
from . import embedded


commands = {
    "matrix": matrix,
    "merge": merge,
    "opcodes": opcodes,
    "platforms": platforms,
    "plot_dataset": plot_dataset,
    "experiments": experiments,
    "summary": summary,
    "compare": compare,
    "figures": figures,
    "tsne": tsne,
    "simulate": simulate,
    "plot_simulation": plot_simulation,
    "embedded": embedded
}
