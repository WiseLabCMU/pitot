"""Prediction experiments."""

import os
import numpy as np
import json
import time
import copy
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser
import optax

from forecast import Dataset, CrossValidationTrainer, presets
from forecast.prediction import models


EPOCHS = 100
EPOCH_SIZE_MULTIPLIER = 250


def _override(default, overrides):
    for k, v in overrides.items():
        _subdict = default
        for subkey in k[:-1]:
            _subdict = _subdict[subkey]
        _subdict[k[-1]] = v
    return default


def create_trainer(dataset, preset="embedding"):
    """Create training manager."""
    config = _override(copy.deepcopy(presets.DEFAULT), presets.PRESETS[preset])
    model = partial(
        getattr(models, config["model"]),
        dataset, shape=dataset.shape, **config["model_args"])
    return config, CrossValidationTrainer(
        dataset, model, optimizer=optax.adam(0.001), **config["training_args"])


def _experiment(name, trainer, p, config={}):
    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    start = time.time()
    # epoch_size = int(EPOCH_SIZE_MULTIPLIER * p)
    epoch_size = 2
    results = trainer.train_replicates(
        p=p, tqdm=pbar, epochs=EPOCHS, epoch_size=epoch_size)

    model_dir = os.path.join("results", name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_dir, "{}.npz".format(p)), **results)

    with open(os.path.join("results", name, "{}.json".format(p)), 'w') as f:
        json.dump({
            "config": config, "duration": time.time() - start,
            "epoch_size": epoch_size, "epochs": EPOCHS, "train_split": p
        }, f)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("methods", nargs="+", default=[])
    p.add_argument("--dataset", "-d", default="data.npz")
    p.add_argument("--interference", "-i", default="if.npz")
    p.add_argument("--no-baseline", dest="baseline", action='store_false')
    p.add_argument("--sparsity", "-s", nargs="+", type=float, default=None)
    p.set_defaults(baseline=True)
    args = p.parse_args()

    if args.sparsity is None:
        args.sparsity = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    dataset = Dataset(
        data=args.dataset, if_data=args.interference, offset=1000. * 1000.)
    for name in args.methods:
        model_dir = os.path.join("results", name)
        config, trainer = create_trainer(dataset, preset=name)
        for s in args.sparsity:
            _experiment(name, trainer, s, config=config)
