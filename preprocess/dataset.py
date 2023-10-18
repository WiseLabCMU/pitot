"""Create dataset from raw collected data."""

import json
import numpy as np
import os
from functools import partial

from beartype.typing import Optional

from prediction.utils import tree_stack
from ._common import workload_name, platform_name, apply_recursive


def _parse(p):
    p.add_argument(
        "-s", "--src", nargs='+', default=[], help="Raw data sources.")
    p.add_argument(
        "-p", "--platforms",
        default="data/_platforms.npz", help="Platform data.")
    p.add_argument(
        "-c", "--opcodes", default="data/_opcodes.npz",
        help="Workload data.")
    p.add_argument("-o", "--out", default="data/data.npz", help="Output path.")
    p.add_argument(
        "--exclude", nargs='+', help="Excluded devices.",
        default=["iwasm-a:hc-21", "iwasm-a:hc-25", "iwasm-a:hc-27"])
    return p


def _load(
    data: dict, devices: dict = {}, pmap: dict = {}, wmap: dict = {},
    exclude: set[str] = set()
) -> Optional[tuple[dict[str, int], float]]:
    t = np.array(data["wall"])
    t = t[t > 0]
    w = workload_name(data["module"])
    device = devices[data["module"]["parent"]]["name"]
    p = platform_name(device, data["module"]["args"]["engine"])

    if len(t) == 0 or p in exclude:
        return None
    else:
        return {"platform": pmap[p], "workload": wmap[w]}, np.mean(t)


def _load2(
    data: dict, pmap: dict = {}, wmap: dict = {}
) -> list[tuple[dict[str, int], float]]:
    pname = pmap[platform_name(data["device"], data["runtime"])]
    return [
        ({"platform": pname, "workload": wmap[workload_name(mod)]}, t)
        for mod, t in data["data"].items()]


def _main(args):

    platform_data = np.load(args.platforms)
    platform_map = {n: i for i, n in enumerate(platform_data["names"])}
    workload_data = np.load(args.opcodes)
    workload_map = {n: i for i, n in enumerate(workload_data["names"])}

    data = []
    for base in args.src:
        if os.path.isdir(base):
            with open(os.path.join(base, "runtimes.json")) as f:
                manifest = json.load(f)
            data += apply_recursive(base, partial(
                _load, devices=manifest, pmap=platform_map, wmap=workload_map,
                exclude=set(args.exclude)))
        else:
            with open(base) as f:
                x = json.load(f)
                for v in x.values():
                    data += _load2(v, pmap=platform_map, wmap=workload_map)

    # Cut invalid samples
    idx, t = tree_stack(data, _np=np)
    t = t.astype(np.float32)

    mask = (t > 0)
    t = t[mask]
    idx = {k: v[mask].astype(np.uint16) for k, v in idx.items()}

    # Shrink empty rows/cols
    pmask = np.zeros(len(platform_map), dtype=bool)
    pmask[idx["platform"]] = True
    pshrink = np.zeros(len(pmask), dtype=int)
    pshrink[pmask] = np.arange(np.sum(pmask))

    wmask = np.zeros(len(workload_map), dtype=bool)
    wmask[idx["workload"]] = True
    wshrink = np.zeros(len(wmask), dtype=int)
    wshrink[wmask] = np.arange(np.sum(wmask))

    np.savez(
        args.out, t=t,
        i_platform=pshrink[idx["platform"]],
        i_workload=wshrink[idx["workload"]],
        d_platform=platform_data["data"][pmask],
        d_workload=workload_data["data"][wmask],
        n_platform=platform_data["names"][pmask],
        n_workload=workload_data["names"][wmask],
        f_platform=platform_data["labels"],
        f_workload=workload_data["labels"])
