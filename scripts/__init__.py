"""Train/eval scripts."""

from . import split, train, evaluate, embeddings

commands = {
    "split": split,
    "train": train,
    "evaluate": evaluate,
    "embeddings": embeddings
}
