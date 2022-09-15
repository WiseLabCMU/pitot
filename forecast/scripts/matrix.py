"""Generate runtimes x modules matrix."""

import numpy as np
from matplotlib import pyplot as plt

from forecast.dataset import Session
from libsilverline import ArgumentParser


def _matrix(path="data", key="cpu_time", suffix="", out=""):
    """Create execution time matrix.

    Parameters
    ----------
    path : str[]
        File paths containing session.
    key : str
        Key to create matrix for.
    suffix : str
        Suffix to attach to the path; used to denote different keys.
    out : str
        Path to save plot to; if blank, uses the same path as the dataset.
    """
    if out == "":
        out = path
    if suffix != "":
        out = "{}_{}".format(out, suffix)

    stats = Session(path).matrix(key=key, save="{}.npz".format(out))

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
    fig.savefig("{}.png".format(out), dpi=100)


def _parse():
    p = ArgumentParser("Generate execution time matrix.")
    p.add_to_parser("matrix", _matrix, group="matrix")
    return p


def _main(args):
    _matrix(**args["matrix"])
