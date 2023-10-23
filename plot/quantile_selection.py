"""Optimal quantile example."""

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

fig, axs = plt.subplots(1, 1, figsize=(5, 3))
yy = np.load("summary/conformal/optimal.npz")['mf_width_raw']
axs.plot(np.arange(8), 100 * yy[4, :, 5, :].T)
axs.yaxis.set_major_formatter(PercentFormatter())
axs.set_xticks(np.arange(8))
axs.set_xticklabels(["50%", "60%", "70%", "80%", "90%", "95%", "98%", "99%"])
axs.grid()
axs.set_ylabel("Bound Tightness")
axs.set_xlabel("Quantile regression target quantile $\\xi$")

fig.tight_layout()
fig.savefig("figures/optimal_quantile.pdf")
