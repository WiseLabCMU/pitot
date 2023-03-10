"""Summarize experiments."""

import os
import json
import jax
import numpy as np
from tqdm import tqdm

from prediction import Objective, Dataset


_desc = "Summarize experiment results."


def _parse(p):
    p.add_argument("-p", "--path", default="results", help="Input directory.")
    return p


def _load(path):
    with open(path.replace(".npz", ".json")) as f:
        config = json.load(f)

    _obj = config["config"]["objectives"]
    dataset = Dataset.from_npz(config["dataset"], log=_obj[0]["log"])
    objectives = [Objective.from_config(dataset, c) for c in _obj]
    result = np.load(path)

    return {
        cfg["name"]: jax.vmap(obj.mape)(
            result[cfg["save"] if cfg["save"] else "C_hat"],
            result[cfg["name"] + "_test"]).tolist()
        for obj, cfg in zip(objectives, _obj)
    }


def _recurse(path):
    contents = os.listdir(path)
    if any(c.endswith(".npz") for c in contents):
        results = [x for x in os.listdir(path) if x.endswith(".npz")]
        results = {
            float(x.replace(".npz", "")): _load(os.path.join(path, x))
            for x in tqdm(results, desc=path)
        }

        with open(os.path.join(path, "summary.json"), 'w') as f:
            json.dump(results, f, indent=4)
    else:
        for c in contents:
            _recurse(os.path.join(path, c))


def _main(args):
    _recurse(args.path)
