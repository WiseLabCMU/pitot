"""Prediction experiments."""

import os
import numpy as np
import json
import time
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
        "layers": [64, 64], "dim": 4, "scale": 0.01},
    "training_args": {
        "beta": (1.0, 0.0), "batch": (128, 0), "replicates": 50,
        "k": 10, "do_baseline": True
    }
}

EPOCHS = 100
EPOCH_SIZE_MULTIPLIER = 250


def _cfg_linear(dim):
    return {
        ("model",): "linear",
        ("model_args",): {"dim": dim, "alpha": 0.001, "scale": 0.01}
    }


def _cfg_if(s):
    return {
        ("model",): "interference", ("model_args", "s"): s,
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    }


PRESETS = {
    # Ablations - Embedding Dimension (r=64)
    # p = 0.2, 0.4, 0.6, 0.8
    "embedding_r8": {("model_args", "layers"): [64, 8]},
    "embedding_r16": {("model_args", "layers"): [64, 16]},
    "embedding_r32": {("model_args", "layers"): [64, 32]},
    "embedding_r64": {("model_args", "layers"): [64, 64]},
    "embedding_r128": {("model_args", "layers"): [64, 128]},
    # Ablations - Embedding Dimension, Linear (r=64)
    # p = 0.1, 0.2, ... 0.9
    "linear_r2": _cfg_linear(2),
    "linear_r4": _cfg_linear(4),
    "linear_r8": _cfg_linear(8),
    "linear_r16": _cfg_linear(16),
    "linear_r32": _cfg_linear(32),
    "linear_r64": _cfg_linear(64),
    "linear_r128": _cfg_linear(128),
    # Ablations - Embedding Input Dimension (q=?)
    # p = 0.2, 0.4, 0.6, 0.8
    "embedding_q2": {("model_args", "dim"): 2},
    "embedding_q4": {("model_args", "dim"): 4},
    "embedding_q8": {("model_args", "dim"): 8},
    "embedding_q16": {("model_args", "dim"): 16},
    "embedding_q32": {("model_args", "dim"): 32},
    # Ablations - Interference Types (s=?)
    # p = 0.2, 0.4, 0.6, 0.8
    "interference_s1": _cfg_if(1),
    "interference_s2": _cfg_if(2),
    "interference_s4": _cfg_if(3),
    "interference_s4": _cfg_if(4),
    # Full experiments
    # p = 0.1, 0.2, ... 0.9
    "interference": {
        ("model",): "interference",
        ("model_args", "s"): 3,
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    },
    "embedding_ignore": {
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    },
    "device_only": {("model_args", "X_m"): None},
    "module_only": {("model_args", "X_d"): None},
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
    return config, CrossValidationTrainer(
        dataset, model, optimizer=optax.adam(0.001), **config["training_args"])


def _experiment(name, trainer, p, config={}):
    pbar = partial(tqdm, desc="{} : {}".format(name, p))
    start = time.time()
    epoch_size = int(EPOCH_SIZE_MULTIPLIER * p)
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
        config, trainer = create_trainer(dataset, preset=name)
        for s in args.sparsity:
            _experiment(name, trainer, s, config=config)
