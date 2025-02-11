"""t-SNE scatter plot visualizations."""

import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

embeddings = np.load("summary/_embeddings.npz")
data = np.load("data/data.npz")


def _tsne(npz, key, seed=42):
    X = np.swapaxes(npz[key], 0, 1).reshape(npz[key].shape[1], -1)
    scaled = StandardScaler().fit_transform(X)
    return TSNE(n_components=2, random_state=seed).fit_transform(scaled)


def _notick(fig, ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(visible=True)
    ax.legend()
    for axis in [ax.xaxis, ax.yaxis]:
        for tick in axis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    fig.tight_layout()


suites = {
    "cortex": 0, "mibench": 1, "polybench": 2,
    "vision": 3, "libsodium": 4, "python": 5
}
display = {
    "cortex": "Cortex",
    "mibench": "Mibench",
    "polybench": "Polybench",
    "vision": "SDVBS",
    "libsodium": "Libsodium",
    "python": "Python"
}

suite_index = np.array(
    [suites[re.split(r'[/:]', x)[0]] for x in data['n_workload']])

tsne = _tsne(embeddings, "Xw",  seed=46)
markers = ['.', '+', '1', 'x', '2', '*']
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for (k, v), m in zip(suites.items(), markers):
    ax.scatter(
        *tsne[suite_index == v].T, c='C{}'.format(v),
        label=display[k], marker=m)
_notick(fig, ax)
fig.savefig("figures/tsne_workload.pdf")


runtimes = {k: i for i, k in enumerate([
    "iwasm-i", "wasmedge-i", "wasm3-i",
    "iwasm-a", "wasmtime-a", "wasmtime-j",
    "wasmer-a-ll", "wasmer-a-cl", "wasmer-j-cl", "wasmer-j-sp"
])}

display = {
    "iwasm-i": "WAMR (interpreted)",
    "wasmedge-i": "WasmEdge (interpreted)",
    "wasm3-i": "Wasm3 (interpreted)",
    "iwasm-a": "WAMR (LLVM AOT)",
    "wasmtime-j": "Wasmtime (Cranelift JIT)",
    "wasmtime-a": "Wasmtime (Cranelift AOT)",
    "wasmer-j-cl": "Wasmer (Cranelift JIT)",
    "wasmer-a-ll": "Wasmer (LLVM AOT)",
    "wasmer-j-sp": "Wasmer (Singlepass JIT)",
    "wasmer-a-cl": "Wasmer (Cranelift AOT)",
}

runtime_index = np.array([
    runtimes[x.split(':')[0]] for x in data["n_platform"]])

tsne = _tsne(embeddings, "Xp", seed=40)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
markers = ['.', 'x', '+', '*', '|', '_', '1', '2', '3', '4']
for (k, v), m in zip(runtimes.items(), markers):
    ax.scatter(
        *tsne[runtime_index == v].T, c='C{}'.format(v),
        label=display[k], marker=m)
_notick(fig, ax)
fig.savefig("figures/tsne_runtime.pdf")


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

architecture_index = np.ones(len(data["n_platform"])) * -1
for i, (k, v) in enumerate(architectures.items()):
    members = np.array([
        i for i, val in enumerate(data["n_platform"])
        if val.split(':')[1] in v])
    architecture_index[members] = i

tsne = _tsne(embeddings, "Xp", 40)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
markers = ['.', 'x', '+', '1', '2']
for (i, k), m in zip(enumerate(architectures), markers):
    ax.scatter(
        *tsne[architecture_index == i].T, c='C{}'.format(i),
        label=k, marker=m)
_notick(fig, ax)
fig.savefig("figures/tsne_device.pdf")
