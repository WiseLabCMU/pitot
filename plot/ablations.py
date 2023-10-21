"""Method ablations."""

import numpy as np
from matplotlib import pyplot as plt
import plot


def _plot_ablations(
    fmt: str, experiments: dict[str, str], linestyle=['-', '--', ':', '-.']
):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    data = {x: dict(np.load(fmt.format(x))) for x in experiments}

    for (x, v), ls in zip(experiments.items(), linestyle):
        y = data[x]["mf_mape"]
        y2 = (
            data[x]["if2_mape"] + data[x]["if3_mape"]
            + data[x]["if4_mape"]) / 3
        plot.plot_errorbar(axs[0], np.arange(9), y, label=v, linestyle=ls)
        plot.plot_errorbar(axs[1], np.arange(9), y2, label=v, linestyle=ls)

    axs[-1].legend()
    axs[0].set_title("Without Interference")
    axs[1].set_title("With Interference")
    axs[0].set_ylabel("Mean Absolute Percent Error")
    for ax in axs:
        plot.format_xsplits(ax)
        ax.set_xlabel("Training Data")
    return fig, axs


fig, axs = _plot_ablations(
    "summary/components/{}.npz", {
        "full": "Log-Residual Objective", "nobaseline": "Log Objective",
        "naiveloss": "Naive Proportional Loss"})
fig.tight_layout()
fig.savefig("figures/ablation_objective.pdf")

fig, axs = _plot_ablations(
    "summary/features/{}.npz", {
        "all": "All Features", "noworkload": "Platform Features Only",
        "noplatform": "Workload Features Only", "blackbox": "No Features"})
axs[0].set_ylim(5, 15.5)
axs[1].set_ylim(8.5, 17)
fig.tight_layout()
fig.savefig("figures/ablation_features.pdf")

fig, axs = _plot_ablations(
    "summary/components/{}.npz", {
        "full": "Interference-Aware",
        "discard": "Discard", "ignore": "Ignore"})
fig.tight_layout()
fig.savefig("figures/ablation_interference.pdf")

fig, axs = _plot_ablations(
    "summary/components/{}.npz", {
        "full": "With Activation Function",
        "notrectified": "Simple Multiplicative"})
fig.tight_layout()
fig.savefig("figures/ablation_relu.pdf")
