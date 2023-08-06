"""Command line tools."""

from . import plot_dataset
from . import compare
from . import figures
from . import tsne
from . import embedded
from . import marginals


commands = {
    "plot_dataset": plot_dataset,
    "compare": compare,
    "figures": figures,
    "tsne": tsne,
    "embedded": embedded,
    "marginals": marginals
}
