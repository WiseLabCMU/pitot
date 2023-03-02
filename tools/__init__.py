"""Command line tools."""

from . import matrix
from . import opcodes
from . import platforms
from . import merge

commands = {
    "matrix": matrix,
    "merge": merge,
    "opcodes": opcodes,
    "platforms": platforms
}
