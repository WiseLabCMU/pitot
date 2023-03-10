"""Run experiments."""

import os
from functools import partial
from tqdm import tqdm
import json
import time
import numpy as np

from prediction import (
    Dataset, Objective, CrossValidationTrainer, presets, models)


_desc = "Run prediction experiment(s)."


def _parse(p):
    p.add_argument("methods", nargs="+", default=[])
    p.add_argument(
        "-o", "--out", default="results",
        help="Directory to place results in.")
    p.add_argument(
        "-d", "--dataset", default="data.npz", help="Dataset to use.")
    p.add_argument(
        "-s", "--sparsity", nargs="+", type=float,
        default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        help="Sparsity levels to run each method on. Defaults to "
        "[0.1, 0.2, ... 0.9].")
    p.add_argument(
        "-k", "--checkpoints", type=int, default=100,
        help="Number of \"epochs\" for accounting (validation & checkpoint).")
    p.add_argument(
        "-n", "--steps", type=int, default=250 * 100,
        help="Training steps; trains for `steps * sparsity` gradient descent "
        "steps, split across `checkpoints` accounting periods.")
    p.add_argument("--key", type=int, default=42, help="PRNG key seed.")


def _create_trainer(dataset, config):
    model = getattr(models, config["model"])(
        dataset, shape=dataset.data.shape, **config["model_args"])
    objectives = [
        Objective.from_config(dataset, cfg) for cfg in config["objectives"]]
    return CrossValidationTrainer(
        dataset, model, objectives, **config["training_args"])


def _experiment(
    name: str, p: float, trainer: CrossValidationTrainer, config: dict, args
):
    pbar = partial(tqdm, desc="{}: {}".format(name, p))
    epoch_size = int(args.steps * p / args.checkpoints)

    start = time.time()
    results = trainer.train_replicates(
        epoch_size=epoch_size, epochs=args.checkpoints, key=42, p=p, tqdm=pbar)

    model_dir = os.path.join(args.out, name)
    os.makedirs(model_dir, exist_ok=True)
    np.savez_compressed(os.path.join(model_dir, "{}.npz".format(p)), **results)

    with open(os.path.join("results", name, "{}.json".format(p)), 'w') as f:
        json.dump({
            "config": config, "duration": time.time() - start,
            "epoch_size": epoch_size, "epochs": args.checkpoints,
            "train_split": p, "dataset": args.dataset
        }, f, indent=4)


def _main(args):

    methods = []
    for method in args.methods:
        if method.endswith('*'):
            methods += presets.match(method)
        else:
            methods.append(method)

    for method in methods:
        config = presets.get_config(method)
        dataset = Dataset.from_npz(
            args.dataset, log=config["objectives"][0]["log"])
        trainer = _create_trainer(dataset, config)
        for s in args.sparsity:
            _experiment(method, s, trainer, config, args)
