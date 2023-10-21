"""Plot interference histograms."""

import numpy as np
from matplotlib import pyplot as plt


mat = np.load("data/data.npz")
baseline = np.zeros((len(mat["n_platform"]), len(mat["n_workload"])))
baseline[mat["i_platform"], mat["i_workload"]] = mat["t"]


def _get_interference(path):
    npz = np.load(path)
    return np.maximum(
        1.0, npz["t"] / baseline[npz["i_platform"], npz["i_workload"]])


ifn = {n: _get_interference("data/if{}.npz".format(n)) for n in [2, 3, 4]}
fig, axs = plt.subplots(3, 1, figsize=(6, 4))
for ax, n in zip(axs, ifn):
    ax.hist(ifn[n], bins=np.linspace(1, 21, 81))
    ax.set_yscale('log')
    ax.set_ylim(0.5, 1e5)
    ax.grid(visible=True)
    ax.set_yticklabels([])
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.text(
        15.5, 5e3, "{}-way Interference".format(n),
        backgroundcolor='white', fontsize=11.0)

for ax in axs[:-1]:
    ax.set_xticklabels([])
axs[-1].set_xlabel("Interference Slowdown")
axs[-1].set_xticklabels(['1x', '2x', '5x', '10x', '20x'])
axs[1].set_ylabel('Log Density')

fig.tight_layout()
fig.savefig('figures/interference.pdf')
