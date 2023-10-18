"""Train model."""

import os
import pickle
import json
import pprint
from tqdm import tqdm
from functools import partial

import numpy as np
from jax import random, clear_caches

from prediction import Split, ObjectiveSet, utils
from pitot import presets, models


def _parse(p):
    p.add_argument("-o", "--out", help="Results directory", default="results")
    p.add_argument(
        "-p", "--preset", default="",
        help="Experiment preset(s); trains presets matching this prefix.")
    p.add_argument(
        "-d", "--presets", default=False, action='store_true',
        help="Show all preset information instead of training.")
    p.add_argument(
        "-s", "--splits", default="splits", help="Data splits directory.")
    return p


def _train(seed, method, cfg, splits_path, out_path, tqdm):
    splits = Split.from_npz(splits_path, method.objectives.objectives)
    state, log = method.train(
        key=seed, splits=splits, tqdm=tqdm, **cfg["training_args"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".pkl", "wb") as f:
        pickle.dump(state, f)
    np.savez(out_path + ".npz", **utils.dict_flatten(log))


def train(args, cfg, path):
    """Run training."""
    # IMPORTANT: clear JAX caches here. Since each method has a different
    # compute graph, and each dataset has slightly different shapes,
    # JAX accumulates JIT artifacts which accumulate in CPU memory.
    # See: https://github.com/google/jax/issues/10828
    clear_caches()

    # Verify config
    os.makedirs(path, exist_ok=True)
    if os.path.exists(os.path.join(path, "config.json")):
        with open(os.path.join(path, "config.json")) as f:
            cfg_existing = json.load(f)
            presets.assert_equal(
                cfg, cfg_existing, xlabel="Selected config",
                ylabel="Config already present at {}".format(path))
    else:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=4)

    # Create method
    objectives = ObjectiveSet.from_config(cfg["objectives"])
    method = models[cfg["model"]].from_config(objectives, **cfg["model_args"])

    # Splits
    for level in sorted(os.listdir(args.splits)):
        split_root = random.PRNGKey(int(float(level) * 1e6))
        split_files = sorted(os.listdir(os.path.join(args.splits, level)))
        split_keys = random.split(split_root, len(split_files))
        # Replicates
        for i, (file, key) in enumerate(zip(split_files, split_keys)):
            out_path = os.path.join(path, level, file.replace(".npz", ""))
            split_path = os.path.join(args.splits, level, file)
            if os.path.exists(out_path + ".pkl"):
                print("Skipping {}: already exists.".format(out_path))
            else:
                pbar = partial(tqdm, desc="{} @ {}: {}/{}".format(
                    path, level, i + 1, len(split_files)))
                _train(
                    random.fold_in(key, cfg["seed"]), method, cfg,
                    split_path, out_path, pbar)


def _main(args):
    if args.presets:
        subset = {p: presets.PRESETS[p] for p in presets.match(args.preset)}
        pprint.pprint(subset)
        exit(0)

    matches = presets.match(args.preset)
    if len(matches) == 0:
        print("No matches found: {}".format(args.preset))
        exit(-1)
    else:
        print("Found {} matching preset(s).".format(len(matches)))
    for preset in matches:
        path = os.path.join(args.out, preset)
        train(args, presets.get_config(preset), path)
