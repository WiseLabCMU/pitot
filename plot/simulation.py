import json
import os
import numpy as np
from matplotlib import pyplot as plt


_desc = "Plot simulation results."


def _parse(p):
    return p


def _load_file(path):
    with open(path) as f:
        data = json.load(f)
    return {
        "mean": np.mean(data['mean']),
        "stderr": np.sqrt(np.var(data["mean"]) / len(data["mean"])),
        "l2": data["layer2"], "l1": data["layer1"]
    }


def _filename(i, p):
    return  "j={},n=200,s={},p={}.json".format(i * 100, i * 0.01, p)


def load_data(method, path="results-simulation", p=5):
    data = [_load_file(os.path.join(path, method, _filename(i + 1, p)))
    for i in range(10)]

    return {k: np.array([x[k] for x in data]) for k in data[0]}


METHODS = {
    "oracle": "Oracle",
    "pitot": "Pitot",
    "mlp": "Single Network",
    "baseline": "Linear Scaling",
    "paragon": "Paragon"
}
STYLES = ['-', ':', '--', '-.', (0, (1, 5))]
N_JOBS = (np.arange(10) + 1) * 100


def _plot_mean(ax, p):
    data = {m: load_data(m, p=p) for m in METHODS}
    for (name, label), fmt in zip(METHODS.items(), STYLES):
        ax.errorbar(
            N_JOBS, data[name]["mean"], yerr=data[name]["stderr"] * 2,
            capsize=2, label=label, linestyle=fmt, marker='.')
    ax.grid()
    ax.set_xlabel("Number of Jobs")


def _plot_layer2(ax, p):
    data = {m: load_data(m, p=p) for m in METHODS}
    for (name, label), fmt in zip(METHODS.items(), STYLES):
        ax.plot(N_JOBS, data[name]["l2"], marker='.', linestyle=fmt, label=label)

    ax.grid()
    ax.set_xlabel("Number of Jobs")


def _main(args):

    fig, axs = plt.subplots(1, 2, figsize=(6, 3.5))
    _plot_mean(axs[0], 5)
    _plot_mean(axs[1], 1)

    axs[0].set_ylabel("Relative Latency")
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc='lower center', ncol=5,
        bbox_to_anchor=(0.5, 0, 0., 0), columnspacing=1.0)
    axs[0].set_title("5% Margin")
    axs[1].set_title("1% Margin")
    fig.tight_layout(pad=0.5, rect=(0.02, 0.1, 0.98, 1))
    fig.savefig("figures/simulation_a.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(6, 3.5))
    _plot_layer2(axs[0], 5)
    _plot_layer2(axs[1], 1)
    axs[0].set_ylabel("Net Utilization of Edge Nodes")
    axs[0].set_title("5% Margin")
    axs[1].set_title("1% Margin")
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc='lower center', ncol=5,
        bbox_to_anchor=(0.5, 0, 0., 0), columnspacing=1.0)

    for ax in axs:
        ax.set_ylim(-0.05, 0.7)
    fig.tight_layout(pad=0.5, rect=(0.02, 0.1, 0.98, 1))
    fig.savefig("figures/simulation_b.pdf")
