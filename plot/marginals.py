"""Error by suite/runtime/device."""

import json
import numpy as np
from jax import numpy as jnp
from jax import vmap
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from prediction import Dataset, Objective


_desc = "Plot mean absolute percent error by suite/runtime/device."


def _parse(p):
    return p


devices = [
    "hc-12", "hc-17", "hc-11", "hc-19",
    "hc-31", "hc-33", "hc-15", "hc-10",
    "hc-13", "hc-14", "hc-16", "hc-42",

    "hc-21", "hc-20", "hc-23",
    "hc-24", "hc-26", "hc-29", "hc-25",
    "hc-27", "hc-43", "hc-28", "hc-80"
]
device_labels = [
    "Intel i7-8650U", "Intel i3-4010U", "Intel i7-4770TE", "Intel x5-Z8330",
    "Intel i5-1145G7", "Intel i7-1165G7", "Intel N4020", "AMD R5-5650G",
    "AMD R5-4500U", "AMD R3-3200U", "AMD A6-1450", "SiFive U74",
    "Raspberry Pi 4", "Raspberry Pi 3B+", "Odroid N2+",
    "Banana Pi M5", "Le Potato", "Odroid C4", "Rock Pro 64",
    "Rock Pi 4b", "Renegade", "Orange Pi 3", "Nucldeo-F767ZI"
]

runtimes = [
    "iwasm", "iwasm-aot", "wasmedge", "wasmtime", "wasmer-singlepass",
    "wasmer-cranelift", "wasmer-llvm", "native"]
runtime_labels = [
    "WAMR (Interpreted)", "WAMR (LLVM)", "WasmEdge", "Wasmtime",
    "Wasmer (Singlepass)", "Wasmer (Cranelift)", "Wasmer (LLVM)", "Native"]

suites = ["cortex", "mibench", "polybench", "vision", "libsodium", "apps"]
suite_labels = [
    "Cortex", "Mibench", "Polybench", "SDVBS", "Libsodium", "Python"]


def _main(args):

    dataset = "data/data.npz"
    ds = Dataset.from_npz(dataset)

    method = "results/embedding/128/0.5"
    with open(method + ".json") as f:
        config = json.load(f)

    _obj = config["config"]["objectives"][0]
    objective = Objective.from_config(ds, _obj)

    result = np.load(method + ".npz")
    pred = jnp.mean(result["C_hat"], axis=1)
    actual = ds.data[None, ...]
    percent_err = np.nan_to_num(
        jnp.abs(np.exp(pred - actual) - 1), nan=0.0, posinf=0.0, neginf=0.0)
    
    npz = np.load("data/data.npz")
    platform = npz["platform"]
    module = npz["module"]

    ij = objective.x[result["mf_test"]]
    test_mask = vmap(ds.to_mask)(ij)

    def apply_mask(mask):
        err_masked = (test_mask * percent_err)[None, ...] * mask
        acc = np.sum(err_masked, axis=(1, 2, 3))
        count = np.sum(test_mask[None, ...] * mask, axis=(1, 2, 3))
        return acc / count
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4))
    x1 = np.arange(len(runtimes))

    runtime_mask = np.char.startswith(
        platform.reshape(1, -1), np.array(runtimes).reshape(-1, 1)
    ).reshape(len(runtimes), 1, -1, 1)
    ax.bar(x1, apply_mask(runtime_mask))

    x2 = np.arange(len(devices)) + x1[-1] + 2
    device_mask = np.char.endswith(
        platform.reshape(1, -1), np.array(devices).reshape(-1, 1)
    ).reshape(len(devices), 1, -1, 1)
    ax.bar(x2, apply_mask(device_mask))

    x3 = np.arange(len(suites)) + x2[-1] + 2
    suite_mask = np.char.startswith(
        module.reshape(1, -1), np.array(suites).reshape(-1, 1)
    ).reshape(len(suites), 1, 1, -1)
    ax.bar(x3, apply_mask(suite_mask))

    ax.set_xticks(np.concatenate([x1, x2, x3], axis=0))
    ax.set_xticklabels(
        runtime_labels + device_labels + suite_labels, rotation=90)
    ax.set_xlim(-1, x3[-1] + 1)
    ax.grid(axis='y')
    ax.axvline((x1[-1] + x2[0]) / 2, color='black', linewidth=0.5)
    ax.axvline((x2[-1] + x3[0]) / 2, color='black', linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Mean Absolute Percent Error")
    fig.tight_layout(pad=0.5)
    fig.savefig("figures/marginals.pdf")
