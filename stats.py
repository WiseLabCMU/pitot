"""Generate statistics npz file."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from dataset import Session
from parse import ArgumentParser


def _matrix(path):
    stats = Session(path).matrix(save="{}.npz".format(path))

    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    axs[0].imshow(np.log(stats["mean"]))
    axs[1].imshow(stats["mean"])
    axs[2].imshow(stats["std"])

    axs[0].set_title("Log Mean Runtime")
    axs[1].set_title("Mean Runtime")
    axs[2].set_title("Stddev")

    for ax in axs:
        ax.set_xticks(np.arange(len(stats["runtimes"])))
        ax.set_xticklabels(stats["runtimes"], rotation='vertical')

    axs[0].set_yticks(np.arange(len(stats['files'])))
    axs[0].set_yticklabels(
        [f.split("/")[-1].split(".")[0] for f in stats["files"]])
    axs[1].set_yticks([])
    axs[2].set_yticks([])

    fig.tight_layout(w_pad=0, h_pad=0)
    fig.savefig("{}.png".format(path), dpi=100)


def _table(path):
    return Session(path).summary(save=path + ".csv")


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument(
        "path", nargs='+', default=["data/polybench"],
        help="Directories containing data to summarize.")
    p.add_argument(
        "--mode", default="matrix", help="Type to compute; matrix or table.")
    args = p.parse_args()

    if args["mode"] == "matrix":
        for path in args["path"]:
            _matrix(path)
    elif args["mode"] == "table":
        for path in args["path"]:
            _table(path)
