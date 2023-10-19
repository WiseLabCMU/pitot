"""Train/eval scripts."""

from . import split, train, evaluate

commands = {
    "split": split,
    "train": train,
    "evaluate": evaluate
}
