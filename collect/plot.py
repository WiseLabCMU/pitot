"""Draw poster-style dataset plot."""

import numpy as np
from matplotlib import pyplot as plt


def _parse(p):
    p.add_argument("-s", "--src", default="data/data.npz", help="Data file.")
    p.add_argument(
        "-o", "--out", default="data/dataset.pdf", help="Output file.")


def _main(args):

    data = np.load(args.src)
    arr = np.zeros((data["d_platform"].shape[0], data["d_workload"].shape[0]))
    arr[data["i_platform"], data["i_workload"]] = data["t"]

    fig, axs = plt.subplots(
        2, 2, figsize=(40, 60),
        gridspec_kw={"height_ratios": [4, 3], "width_ratios": [1, 4.5]})

    with np.errstate(divide='ignore'):
        axs[0, 1].imshow(np.log(arr), aspect='auto')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

    axs[0, 0].imshow(data["d_platform"])
    axs[0, 0].set_xticks(np.arange(data["d_platform"].shape[1]))
    axs[0, 0].set_xticklabels(data["f_platform"], rotation=90)
    axs[0, 0].set_yticks(np.arange(data["d_platform"].shape[0]))
    axs[0, 0].set_yticklabels(data["n_platform"])

    axs[1, 1].imshow(data["d_workload"].T)
    axs[1, 1].set_yticks(np.arange(data["d_workload"].shape[1]))
    axs[1, 1].set_yticklabels(data["f_workload"])
    axs[1, 1].set_xticks(np.arange(data["d_workload"].shape[0]))
    axs[1, 1].set_xticklabels(data["n_workload"], rotation=90)

    axs[1, 0].axis('off')
    fig.tight_layout(h_pad=-8, w_pad=1)

    fig.savefig(args.out)
