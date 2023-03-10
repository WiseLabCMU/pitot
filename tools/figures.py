"""Draw figures."""

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

from jax import numpy as jnp
from jax import vmap


_desc = "Miscellaneous paper figures."


def _parse(p):
    p.add_argument("figures", nargs='+', default=[], help="Figures to draw.")
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path", help="Results directory.", default="results")
    p.add_argument(
        "-d", "--data", help="Dataset file.", default="data/data.npz")
    p.add_argument(
        "--if_data", help="Interference dataset.", default="data/data.if.npz")
    p.add_argument("-i", "--dpi", help="Out image DPI.", default=400, type=int)
    return p


class Figures:
    """Paper figures."""

    @staticmethod
    def matrix(args):
        """Matrix dataset 'thumbnail' picture."""
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        with np.errstate(divide='ignore'):
            data = np.log(np.load(args.data)["data"])
        axs.imshow(data)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_ylabel("Platforms")
        axs.set_xlabel("Modules")
        fig.tight_layout()

        return {"matrix": fig}

    @staticmethod
    def spectral_norm(args):
        """Interference spectral norm interpretation."""
        ds = np.load(args.if_data)
        data = np.load(args.path + "/interference/1/0.9.npz")

        baseline = ds["data"][ds["if_platform"], ds["if_module"]]
        interference = ds["if_data"] / baseline - 1

        measured = [
            np.mean(interference[ds["if_platform"] == i])
            for i in range(np.max(ds["if_platform"]) + 1)]

        def _spectral_norm(v_s, v_g):
            return jnp.linalg.norm(jnp.matmul(v_s, v_g.T))

        vvv = vmap(vmap(vmap(_spectral_norm)))
        pred = np.mean(vvv(data["V_s"], data["V_g"]), axis=(0, 1))

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.grid()
        ax.scatter(measured, pred)
        ax.set_ylabel(r"Learned: $\mathbb{E}[||\mathbf{F}_j||_2]$")
        ax.set_xlabel("Measured: Mean Percent Interference")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig.tight_layout()
        return {"spectral_norm": fig}


def _main(args):
    for name in args.figures:
        figs = getattr(Figures, name)(args)
        for k, v in figs.items():
            out = os.path.join(args.out, k + ".png")
            print("Created:", out)
            v.savefig(out, dpi=args.dpi)
