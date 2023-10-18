"""Assemble interference dataset."""

from functools import partial
import json
import numpy as np
import os

from jaxtyping import Float

from prediction.utils import tree_stack
from ._common import workload_name, platform_name, apply_recursive


def _parse(p):
    p.add_argument(
        "-s", "--src", nargs='+', default=[], help="Raw data sources.")
    p.add_argument(
        "-d", "--dataset",
        default="data/data.npz", help="Main non-interference dataset.")
    p.add_argument("-o", "--out", default="data/if2.npz", help="Output path.")
    return p


def _load(
    data: dict, baseline: Float[np.ndarray, "..."], devices: dict = {},
    pmap: dict = {}, wmap: dict = {}
) -> list[tuple[dict[str, int], float]]:
    w = workload_name(data["module"])
    device = devices[data["module"]["parent"]]["name"]
    p = platform_name(device, data["module"]["args"]["engine"])

    ww_raw = data["module"]["file"].split(":")
    ww = list(map(workload_name, ww_raw))
    p = platform_name(device, data["module"]["args"]["engine"])

    res = []
    for i, (w, wr) in enumerate(zip(ww, ww_raw)):
        t = np.array(data["{}:{}".format(i, wr)])
        if np.any(t[:-1] == 0):
            return []

        t = t[t > 0]
        try:
            if baseline[pmap[p], wmap[w]] > 0 and len(t) > 0:
                interferers = [wj for j, wj in enumerate(ww) if j != i]
                idx = {
                    "interference{}".format(j): wmap[wj]
                    for j, wj in enumerate(interferers)}
                res.append((
                    {"platform": pmap[p], "workload": wmap[w], **idx},
                    np.mean(t)))
        except KeyError:
            pass
    return res


def _main(args):

    npz = np.load(args.dataset)

    platform_map = {n: i for i, n in enumerate(npz["n_platform"])}
    workload_map = {n: i for i, n in enumerate(npz["n_workload"])}

    baseline = np.zeros((len(platform_map), len(workload_map)))
    baseline[npz["i_platform"], npz["i_workload"]] = npz["t"]

    # Super janky loading code
    data = []
    for base in args.src:
        with open(os.path.join(base, "runtimes.json")) as f:
            manifest = json.load(f)
        data += apply_recursive(base, partial(
            _load, baseline=baseline, devices=manifest,
            pmap=platform_map, wmap=workload_map))

    idx, t = tree_stack(sum(data, []), _np=np)
    interference = {
        "i_{}".format(k): v
        for k, v in idx.items() if k.startswith("interference")}

    np.savez(
        args.out, t=t, i_platform=idx["platform"], i_workload=idx["workload"],
        **interference)
