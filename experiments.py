"""Prediction experiments."""

import os
import numpy as np
from tqdm import tqdm
from jax import random, vmap
from functools import partial

from dataset import Dataset
from prediction import CrossValidationTrainer
from prediction.models import *


def _experiment(model, constr, kwargs, s):
    trainer = CrossValidationTrainer(
        ds, partial(constr, name=model, **kwargs), optimizer=constr.optimizer,
        epochs=constr.epochs, epoch_size=constr.epoch_size, batch=64,
        tqdm=partial(tqdm, desc="{} : {}".format(model, s)))
    results = trainer.train_replicates(key, replicates=20, p=s, k=24)
    results_exp = vmap(trainer.export_results)(results)
    np.savez_compressed(
        os.path.join(model_dir, "{}.npz".format(s)), **results_exp)


ds = Dataset("polybench.npz", offset=10000., key=lambda x: x['percentile'][95])
_kw = {"scale": 10.0, "samples": ds.matrix.shape}
_pca = ds.opcodes_pca

MODELS = {
    "rank1": (MFLinear, {"dim": 1, **_kw}),
    "rank8": (MFLinear, {"dim": 8, **_kw}),
    "rank32": (MFLinear, {"dim": 32, **_kw}),
    "nn8": (MFNN, {"dim": (8, 8), **_kw}),
    "nn32": (MFNN, {"dim": (32, 32), **_kw}),
    "opcodes8": (MFNNSI, {"side_info": ds.opcodes, "dim": (0, 8), **_kw}),
    "opcodes32": (MFNNSI, {"side_info": ds.opcodes, "dim": (0, 32), **_kw}),
    "pca8": (MFNNSI, {"side_info": _pca[:, :8], "dim": (0, 8), **_kw}),
    "pca32": (MFNNSI, {"side_info": _pca[:, :32], "dim": (0, 32), **_kw}),
    "both8": (MFNNSI, {"side_info": _pca[:, :8], "dim": (8, 8), **_kw}),
    "both32": (MFNNSI, {"side_info": _pca[:, :32], "dim": (32, 32), **_kw}),
    "linearopcodes": (LinearModel, {"features": ds.opcodes, **_kw}),
    "linearpca": (LinearModel, {"features": _pca[:, :32], **_kw}),
    "embedding": (Embedding, {"side_info": _pca[:, :32], **_kw})
}

SPARSITY = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


if __name__ == "__main__":

    key = random.PRNGKey(42)

    for model, (constr, kwargs) in MODELS.items():
        model_dir = os.path.join("results", model)
        os.makedirs(model_dir, exist_ok=True)

        for s in SPARSITY:
            _experiment(model, constr, kwargs, s)
