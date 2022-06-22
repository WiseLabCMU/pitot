"""Generate statistics npz file."""

import numpy as np
from matplotlib import pyplot as plt


from dataset import Session
from parse import ArgumentParser


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument(
        "path", default="data/polybench",
        help="Directory containing data to summarize.")
    args = p.parse_args()

    stats = Session(args["path"]).stats(save="{}.npz".format(args["path"]))

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
    fig.savefig("{}.png".format(args["path"]), dpi=100)
