"""Generate statistics npz file."""

import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from dataset import Session


def _matrix(path, key="cpu_time", suffix=None, out=None):
    if out is None:
        out = path
    if suffix is not None:
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


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument(
        "path", nargs='+', default=["data/polybench"],
        help="Directories containing data to summarize.")
    p.add_argument("--out", default=None, help="Path to save to.")
    p.add_argument("--suffix", default=None, help="Save path suffix.")
    p.add_argument("--key", default="cpu_time", help="Statistic of interest.")
    args = p.parse_args()

    for path in args.path:
        _matrix(path, out=args.out, key=args.key, suffix=args.suffix)
