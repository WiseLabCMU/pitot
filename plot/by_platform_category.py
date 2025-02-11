"""Error by type of platform."""

import json
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(sys.path[0], ".."))

from pitot import models
from prediction import ObjectiveSet, Split, loss

err_platform = {}
err_workload = {}


def load(path):

    with open(os.path.join("results/pitot", "config.json")) as f:
        cfg = json.load(f)

    objectives = ObjectiveSet.from_config(cfg["objectives"])
    model = models[cfg["model"]].from_config(objectives, **cfg["model_args"])
    q_test = np.arange(90, 100)
    alphas = (100 - q_test) / 100

    splits = Split.from_npz(
        os.path.join("splits/0.9", path), objectives.objectives)
    checkpoint = os.path.join("results/pitot/0.9", path.replace(".npz", ".pkl"))
    with open(checkpoint, 'rb') as f:
        params = pickle.load(f).params

    itest = {'mf': splits['mf'].test}
    indices = model.objectives.index(itest)
    test = model.evaluate(params, indices)
    mape = loss.PercentError(log=True)(test['mf'].y_true, test['mf'].y_hat)

    for ip, iw, err in zip(indices['mf'].x['platform'], indices['mf'].x['workload'], mape):
        ip = int(ip)
        iw = int(iw)
        err = float(err[0])
        if ip not in err_platform:
            err_platform[ip] = []
        if iw not in err_workload:
            err_workload[iw] = []
        err_platform[ip].append(err)
        err_workload[iw].append(err)


for path in ["0.npz", "1.npz", "2.npz", "3.npz", "4.npz"]:
    load(path)



def visualize(ax, cfg, split=0):
    names = np.load("data/data.npz")['n_platform']

    ii = np.array([cfg['key'][x.split(":")[split]] for x in names])

    mean = np.array([np.mean(err_platform[i]) for i in range(len(names))]) * 100
    std = np.array(
        [np.std(err_platform[i]) / np.sqrt(len(err_platform[i]) - 1)
         for i in range(len(names))]) * 100
    order = np.argsort(np.argsort(mean))

    for i, k in enumerate(cfg['description']):
        mask = (ii == k)
        ax.boxplot(mean[mask], positions=[i])
    ax.set_xticks(np.arange(len(cfg['description'])))
    ax.set_xticklabels(cfg['description'].values())
    ax.set_ylim(0, 25)
    ax.yaxis.grid(True)


fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), width_ratios=[3, 4, 5])
visualize(axs[2], {
    "key": {
        "iwasm-a": 0, "iwasm-i": 0,
        "wasm3-i": 1,
        "wasmedge-i": 2,
        "wasmer-a-cl": 3, "wasmer-a-ll": 3, "wasmer-j-cl": 3, "wasmer-j-sp": 3,
        "wasmtime-a": 4, "wasmtime-j": 4
    },
    "description": {0: "WAMR", 1: "Wasm3", 2: "Wasmedge", 3: "Wasmer", 4: "Wasmtime"}
})
visualize(axs[0], {
    "key": {
        "iwasm-a": 0, "iwasm-i": 1,
        "wasm3-i": 1,
        "wasmedge-i": 1,
        "wasmer-a-cl": 0, "wasmer-a-ll": 0, "wasmer-j-cl": 2, "wasmer-j-sp": 2,
        "wasmtime-a": 0, "wasmtime-j": 2
    },
    "description": {0: "AOT", 1: "Interpreted", 2: "JIT"}
})
visualize(axs[1], {
    "key": {
        'hc-10': 0, 'hc-11': 1, 'hc-12': 1, 'hc-13': 0, 'hc-14': 0,
        'hc-15': 1, 'hc-16': 0, 'hc-17': 1, 'hc-18': 1, 'hc-19': 1,
        'hc-20': 2, 'hc-21': 2, 'hc-24': 2, 'hc-25': 2, 'hc-26': 2,
        'hc-27': 2, 'hc-28': 2, 'hc-29': 2, 'hc-31': 1, 'hc-33': 1,
        'hc-34': 1, 'hc-35': 1, 'hc-42': 3, 'hc-43': 2, 'hc-80': 3
    },
    "description": {
        0: 'x86/AMD', 1: 'x86/Intel', 2: 'ARM A-class', 3: 'Other'
    }
}, split=1)
axs[0].set_ylabel("Percent Error")
for ax in axs[1:]:
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.label1.set_visible(False)
fig.tight_layout()
fig.savefig("figures/by_platform_category.pdf")

