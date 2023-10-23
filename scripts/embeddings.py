"""Extract embeddings from a model."""

import json
import pickle
import os
import numpy as np

import pitot
from prediction import ObjectiveSet, utils


def _parse(p):
    p.add_argument("-p", "--path", help="Experiment path.")
    p.add_argument("-o", "--out", help="Output path.")
    return p


def _main(args):

    with open(os.path.join(args.path, "config.json")) as f:
        cfg = json.load(f)

    objectives = ObjectiveSet.from_config(cfg["objectives"])

    # Only implemented for Pitot, but you could also implement this for other
    # matrix factorization-type models.
    assert cfg["model"] == "Pitot"
    model = pitot.Pitot.from_config(objectives, **cfg["model_args"])

    base = os.path.join(args.path, "0.9")
    replicates = sorted([x for x in os.listdir(base) if x.endswith('.pkl')])

    def _load(path):
        with open(os.path.join(base, path), 'rb') as f:
            params = pickle.load(f).params
        return model.extract_embeddings(params)

    embeddings = utils.tree_stack([_load(p) for p in replicates])
    np.savez(args.out, **embeddings)
