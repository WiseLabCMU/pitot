"""Draw figures."""

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

from jax import numpy as jnp
from jax import vmap
from jax import random

from prediction import Dataset, Rank1


_desc = "Miscellaneous paper figures."


def _parse(p):
    p.add_argument(
        "-f", "--figures", nargs='+',
        default=["svd", "matrix", "spectral_norm", "interference_hist"],
        help="Figures to draw.")
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path", help="Results directory.", default="results")
    p.add_argument(
        "-d", "--data", help="Dataset file.", default="data/data.npz")
    p.add_argument(
        "--if_data", help="Interference dataset.", default="data/data.if.npz")
    return p


class Figures:
    """Paper figures."""

    @staticmethod
    def svd(args):
        """Matrix spectrum."""
        ds = Dataset.from_npz(args.data)
        rank1 = Rank1(ds.data, max_iter=1000).fit(ds.to_mask(ds.x))
        X = (ds.data - Rank1.predict(rank1)) * ds.to_mask(ds.x)
        _, S, _ = np.linalg.svd(X)

        X_iid = random.normal(
            random.PRNGKey(42), shape=X.shape
        ) * np.sqrt(np.var(X)) + np.mean(X)
        _, S_iid, _ = np.linalg.svd(X_iid)

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        ax.plot(np.arange(25), S[:25], 'o-', label="Residual Spectrum")
        ax.plot(np.arange(25), S_iid[:25], '--', label="Random Matrix")
        ax.set_yscale('log')
        ax.set_ylabel("Singular value (log-scale)")
        ax.set_xlabel("First 25 Singular Values")
        ax.grid()
        ax.legend()
        fig.tight_layout()
        return {"svd": fig}

    @staticmethod
    def matrix(args):
        """Matrix dataset 'thumbnail' picture."""
        fig, axs = plt.subplots(1, 1, figsize=(5, 4))
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

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        ax.grid()
        ax.scatter(measured, pred, marker='.')
        ax.set_ylabel(r"Learned: $\mathbb{E}[||\mathbf{F}_j||_2]$")
        ax.set_xlabel("Measured: Mean Percent Interference")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig.tight_layout()
        return {"spectral_norm": fig}

    @staticmethod
    def interference_hist(args):
        """Interference histogram."""
        ds = np.load(args.if_data)

        baseline = ds["data"][ds["if_platform"], ds["if_module"]]
        interference = ds["if_data"] / baseline - 1
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        ax.hist(interference, bins=np.linspace(0, 2.50, 51), density=True)
        ax.set_yscale('log')
        ax.set_xlabel("Interference Slowdown")
        ax.set_ylabel("Probability (log-scale)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid()
        fig.tight_layout()
        return {"interference_hist": fig}


def _main(args):
    for name in args.figures:
        figs = getattr(Figures, name)(args)
        for k, v in figs.items():
            out = os.path.join(args.out, k + ".pdf")
            print("Created:", out)
            v.savefig(out)
