"""Create data splits."""

import os
from tqdm import tqdm

import numpy as np
from jax import random

from prediction import ObjectiveSet, utils


OBJECTIVES = {
    "mf": {"path": ["data/data.npz"]},
    "if2": {"path": ["data/data.npz", "data/if2.npz"]},
    "if3": {"path": ["data/data.npz", "data/if3.npz"]},
    "if4": {"path": ["data/data.npz", "data/if4.npz"]}
}


def _parse(p):
    p.add_argument("-o", "--out", default="splits", help="Output directory.")
    p.add_argument("-k", "--seed", default=42, type=int, help="Random seed.")
    p.add_argument(
        "-n", "--replicates", default=5, type=int,
        help="Number of replicates.")
    p.add_argument(
        "-s", "--splits", nargs='+', type=float, help="Split levels.",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return p


def _create_split(
    key: random.PRNGKeyArray, obj: ObjectiveSet, split: float, path: str
) -> None:
    """Create data split."""
    _args = {"train": split * 0.8, "val": split * 0.2}
    split_args = {k: _args for k in obj.objectives}
    splits = obj.split(key, split_args)
    as_dict = utils.dict_flatten({k: v.as_dict() for k, v in splits.items()})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **as_dict)


def _main(args):
    objectives = ObjectiveSet.from_config(OBJECTIVES)
    for split in tqdm(args.splits):
        split_root = random.PRNGKey(int(split * 1e6))
        split_keys = random.split(split_root, args.replicates)
        for i, key in enumerate(split_keys):
            merged = random.fold_in(key, args.seed)
            path = os.path.join(args.out, str(split), "{}.npz".format(i))
            _create_split(merged, objectives, split, path)
