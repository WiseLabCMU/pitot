"""Data collection/marshalling."""

from . import platforms, opcodes, dataset, plot, interference

commands = {
    "platforms": platforms,
    "opcodes": opcodes,
    "dataset": dataset,
    "interference": interference,
    "plot": plot
}
