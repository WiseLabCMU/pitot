"""Draw poster-style dataset plot."""

import numpy as np
from matplotlib import pyplot as plt


def _parse(p):
    p.add_argument("-s", "--src", default="data/data.npz", help="Data file.")
    p.add_argument(
        "-o", "--out", default="data/pitot_12x18.pdf", help="Output file.")


def _main(args):

    data = np.load(args.src)
    arr = np.zeros((data["d_platform"].shape[0], data["d_workload"].shape[0]))
    arr[data["i_platform"], data["i_workload"]] = data["t"]

    fig, axs = plt.subplots(
        # 2, 2, figsize=(40, 60),
        2, 2, figsize=(10, 15),
        gridspec_kw={"height_ratios": [4, 2.0], "width_ratios": [1, 4.5]})

    with np.errstate(divide='ignore'):
        axs[0, 1].imshow(np.log(arr), aspect='auto', interpolation='none')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

    axs[0, 0].imshow(data["d_platform"], aspect='auto', interpolation='none')
    axs[0, 0].set_xlabel("Platform Features $\longrightarrow$", loc='left')
    axs[0, 0].set_ylabel(
        "Platforms (Devices $\\times$ WebAssembly Runtimes) $\longrightarrow$",
        loc='bottom')
    # axs[0, 0].set_xticks(np.arange(data["d_platform"].shape[1]))
    # axs[0, 0].set_xticklabels(data["f_platform"], rotation=90)
    # axs[0, 0].set_yticks(np.arange(data["d_platform"].shape[0]))
    # axs[0, 0].set_yticklabels(data["n_platform"])

    axs[1, 1].imshow(data["d_workload"].T, aspect='auto', interpolation='none')
    axs[1, 1].set_xlabel(
        "Workloads (WebAssembly Modules) $\longrightarrow$", loc='left')
    axs[1, 1].set_ylabel(
        "Workload Features (Opcode Counts) $\longrightarrow$", loc='bottom')
    # axs[1, 1].set_yticks(np.arange(data["d_workload"].shape[1]))
    # axs[1, 1].set_yticklabels(data["f_workload"])
    # axs[1, 1].set_xticks(np.arange(data["d_workload"].shape[0]))
    # axs[1, 1].set_xticklabels(data["n_workload"], rotation=90)

    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    axs[1, 0].axis('off')
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    # fig.tight_layout(h_pad=-8, w_pad=1)
    fig.subplots_adjust(left=0.05, bottom=0.0333, right=0.965, top=0.97666)
    fig.text(0.056, 0.038, "Pitot", fontsize=48, rotation=90, weight='light')
    fig.text(
        0.125, 0.0365, "Bringing Runtime Prediction\nup to speed for the Edge",
        fontsize=18, rotation=90)
    axs[0, 0].text(
        data['d_platform'].shape[1] - 4, 3, "Platforms",
        backgroundcolor=(1, 1, 1, 0.75), rotation=90, fontsize=12, va='top',
        ha='right')
    axs[0, 1].text(
        arr.shape[1] - 5, 3, "Runtime",
        backgroundcolor=(1, 1, 1, 0.75), rotation=90, fontsize=12, va='top',
        ha='right')
    axs[1, 1].text(
        data['d_workload'].shape[0] - 5, 4, "Workloads",
        backgroundcolor=(1, 1, 1, 0.75), rotation=90, fontsize=12, va='top',
        ha='right')

    fig.savefig(args.out)
