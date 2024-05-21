"""Ablation summary table."""

import os
import numpy as np


def _load_result(p):
    npz = np.load(os.path.join("summary", "components", p + ".npz"))
    return (
        npz["mf_mape"],
        (npz["if2_mape"] + npz["if3_mape"] + npz["if4_mape"]) / 3)

def _load_result2(p):
    npz = np.load(os.path.join("summary", "conformal", p + ".npz"))
    return (
        npz["mf_width"][4].T, (
            npz["if2_width"][4].T + npz["if3_width"][4].T
            + npz["if4_width"][4].T) / 3)


def _stats(x, y):
    return (
        100 * np.mean(x - y),
        100 * np.std(x - y, ddof=1) / np.sqrt(x.shape[0]) * 2)


baseline = _load_result("full")
ablations = {
    "Simple Log Objective": "nobaseline",
    "Naive Propotional Loss": "naiveloss",
    "Discard": "discard",
    "Ignore": "ignore",
    "Simple Multiplicative": "notrectified",
}

for k, v in ablations.items():
    res = _load_result(v)
    print("{} & ${:.2f}\% \pm {:.2f}\%$ & ${:.2f}\% \pm {:.2f}\% $".format(
        k, *_stats(res[0], baseline[0]), *_stats(res[1], baseline[1])
    ))

baseline2 = _load_result2("optimal")
ablations2 = {
    "Non-quantile SCR": "nonquantile",
    "Naive CQR": "naive"
}

for k, v in ablations2.items():
    res = _load_result2(v)
    print("{} & ${:.2f}\% \pm {:.2f}\%$ & ${:.2f}\% \pm {:.2f}\% $".format(
        k, *_stats(res[0], baseline2[0]), *_stats(res[1], baseline2[1])
    ))
