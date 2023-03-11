"""Draw figures."""

import os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


_desc = "Draw TSNE figures used in the paper."


def _parse(p):
    p.add_argument("-o", "--out", help="Output directory.", default="figures")
    p.add_argument(
        "-p", "--path", help="Results directory.", default="results")
    p.add_argument(
        "-d", "--data", help="Dataset file.", default="data/data.npz")
    p.add_argument("-i", "--dpi", help="Out image DPI.", default=400, type=int)
    return p


def _tsne(npz, key, dims=64):
    x = np.swapaxes(npz[key], 2, 3)
    x = x.reshape(-1, x.shape[-1]).T

    scaled = StandardScaler().fit_transform(x)
    reduced = PCA(n_components=dims).fit_transform(scaled)
    return TSNE(n_components=2, random_state=42).fit_transform(reduced)


def _plot_p_arch(ax, dataset, P):
    architectures = {
        'AMD x86': {'hc-10', 'hc-13', 'hc-14', 'hc-16'},
        'Intel x86': {
            'hc-11', 'hc-12', 'hc-15', 'hc-17', 'hc-18',
            'hc-19', 'hc-31', 'hc-33', 'hc-34', 'hc-35'},
        'ARM A-class': {
            'hc-20', 'hc-21', 'hc-23', 'hc-24', 'hc-25', 'hc-26', 'hc-27',
            'hc-28', 'hc-28', 'hc-29', 'hc-43'},
        'RISC-V': {'hc-42'},
        'ARM M-class': {'hc-80'}
    }

    architecture_index = np.ones(len(dataset["platform"])) * -1
    for i, (k, v) in enumerate(architectures.items()):
        members = np.array([
            i for i, val in enumerate(dataset["platform"])
            if val.split('.')[1] in v])
        architecture_index[members] = i

    markers = ['.', 'x', '+', '1', '2']
    for (i, k), m in zip(enumerate(architectures), markers):
        ax.scatter(
            *P[architecture_index == i].T, c='C{}'.format(i),
            label=k, marker=m)


def _plot_p_runtime(ax, dataset, P):

    runtimes = {
        "iwasm": 0, "iwasm-aot": 1, "native": 2,
        "wasmedge": 3, "wasmer-cranelift": 4, "wasmer-llvm": 5,
        "wasmer-singlepass": 6, "wasmtime": 7
    }
    display = {
        "iwasm": "WAMR (interpreted)",
        "iwasm-aot": "WAMR (LLVM AOT)",
        "native": "Native",
        "wasmedge": "WasmEdge (interpreted)",
        "wasmer-cranelift": "Wasmer (Cranelift JIT)",
        "wasmer-llvm": "Wasmer (LLVM AOT)",
        "wasmer-singlepass": "Wasmer (Singlepass JIT)",
        "wasmtime": "Wasmtime (JIT)"
    }

    runtime_index = np.array([
        runtimes[x.split('.')[0]] for x in dataset["platform"]])

    markers = ['.', 'x', '+', 'p', '1', '2', '3', 'd']
    for (k, v), m in zip(runtimes.items(), markers):
        ax.scatter(
            *P[runtime_index == v].T, c='C{}'.format(v),
            label=display[k], marker=m)


def _plot_m_suite(ax, dataset, M):

    suites = {
        "cortex": 0, "mibench": 1,
        "polybench/mini": 2, "polybench/small": 3, "polybench/medium": 4,
        "vision": 5
    }
    display = {
        "cortex": "Cortex",
        "mibench": "Mibench",
        "polybench/mini": "Polybench (mini)",
        "polybench/small": "Polybench (small)",
        "polybench/medium": "Polybench (medium)",
        "vision": "SDVBS"
    }

    suite_index = np.array(
        [suites["/".join(x.split('/')[:-1])] for x in dataset["module"]])

    markers = ['.', '1', '2', '3', 'x', '+']
    for (k, v), m in zip(suites.items(), markers):
        ax.scatter(
            *M[suite_index == v].T, c='C{}'.format(v),
            label=display[k], marker=m)


def _main(args):

    dataset = np.load(args.data)
    data = np.load(args.path + "/embedding/128/0.9.npz")
    M = _tsne(data, "M", dims=64)
    P = _tsne(data, "P", dims=64)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    _plot_m_suite(axs[0], dataset, M)
    _plot_p_arch(axs[1], dataset, P)
    _plot_p_runtime(axs[2], dataset, P)
    axs[0].set_title("Module Embedding")
    axs[1].set_title("Platform Embedding (By Architecture)")
    axs[2].set_title("Platform Embedding (By Runtime)")

    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.legend()
        for axis in [ax.xaxis, ax.yaxis]:
            for tick in axis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tsne.png"))
