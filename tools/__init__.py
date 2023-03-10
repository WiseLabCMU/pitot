"""Command line tools."""

from . import matrix
from . import opcodes
from . import platforms
from . import merge
from . import plot_large
from . import experiments
from . import summary
from . import ablations
from . import figures
from . import tsne


commands = {
    "matrix": matrix,
    "merge": merge,
    "opcodes": opcodes,
    "platforms": platforms,
    "plot_large": plot_large,
    "experiments": experiments,
    "summary": summary,
    "ablations": ablations,
    "figures": figures,
    "tsne": tsne
}
