"""Command line tools."""

from . import plot_dataset
from . import compare
from . import figures
from . import tsne
from . import marginals
from . import simulation


commands = {
    "plot_dataset": plot_dataset,
    "compare": compare,
    "figures": figures,
    "tsne": tsne,
    "marginals": marginals,
    "simulation": simulation
}
