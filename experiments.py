"""Prediction experiments."""

import os
import numpy as np
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser
import optax

from forecast import Dataset, CrossValidationTrainer
from forecast.prediction import models


DEFAULT = {
    "model": "embedding",
    "model_args": {
        "X_m": True, "X_d": True, "alpha": 0.001,
        "layers": [64, 32], "dim": 4, "scale": 0.01},
    "training_args": {
        "beta": (1.0, 0.0), "epochs": 250, "epoch_size": 100,
        "batch": (64, 1), "replicates": 100, "k": 20, "do_baseline": True
    }
}

PRESETS = {
    "embedding": {},
    "interference": {
        ("model",): "interference",
        ("model_args", "s"): 3,
        ("training_args", "beta"): (1.0, 0.0),
        ("training_args", "batch"): (64, 64)
    },
    "device_only": {("model_args", "X_m"): None},
    "module_only": {("model_args", "X_d"): None},
    "linear4": {("model_args",): {"alpha": 0.001, "dim": 4, "scale": 0.01}},
    "linear8": {("model_args",): {"alpha": 0.001, "dim": 8, "scale": 0.01}},
    "linear16": {("model_args",): {"alpha": 0.001, "dim": 16, "scale": 0.01}},
    "linear32": {("model_args",): {"alpha": 0.001, "dim": 32, "scale": 0.01}},
    "linear64": {("model_args",): {"alpha": 0.001, "dim": 64, "scale": 0.01}},
    "linear128": {("model_args",): {"alpha": 0.001, "dim": 128, "scale": 0.01}}
}


def _override(default, overrides):
    for k, v in overrides.items():
        _subdict = default
        for subkey in k[:-1]:
            _subdict = _subdict[subkey]
        _subdict[k[-1]] = v
    return default


def create_trainer(dataset, preset="embedding"):
    """Create training manager."""
    config = _override(DEFAULT, PRESETS[preset])
    model = partial(
        getattr(models, config["model"]),
        dataset, shape=dataset.shape, **config["model_args"])
    return CrossValidationTrainer(
        dataset, model, optimizer=optax.adam(0.001), **config["training_args"])


def _experiment(name, trainer, p):
    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    results = trainer.train_replicates(p=p, tqdm=pbar)

    model_dir = os.path.join("results", name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_dir, "{}.npz".format(p)), **results)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("methods", nargs="+", default=[])
    p.add_argument("--dataset", "-d", default="data.npz")
    p.add_argument("--interference", "-i", default=None)
    p.add_argument("--no-baseline", dest="baseline", action='store_false')
    p.add_argument("--sparsity", nargs="+", type=float, default=None)
    p.set_defaults(baseline=True)
    args = p.parse_args()

    if args.sparsity is None:
        args.sparsity = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    dataset = Dataset(
        data=args.dataset, if_data=args.interference, offset=1000. * 1000.)
    for name in args.methods:
        model_dir = os.path.join("results", name)
        trainer = create_trainer(dataset, preset=name)

        for s in args.sparsity:
            _experiment(name, trainer, s)
