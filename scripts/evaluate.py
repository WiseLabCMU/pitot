"""Evaluate model.

NOTE: is generally faster to run CPU-only.
"""

import os
import pickle
import json
from tqdm import tqdm
from functools import partial

import jax
from jax import numpy as jnp
import numpy as np

from prediction import Split, utils, calibrate, loss, ObjectiveSet
from pitot import models


def _parse(p):
    p.add_argument("-p", "--path", help="Experiment path.")
    p.add_argument("-s", "--splits", default="splits", help="Splits path.")
    p.add_argument("-o", "--out", help="Output path.")
    return p


def _metrics(val, test, alphas, log=True):
    margin, v_width = calibrate.calibrate(val.y_true, val.y_hat, alpha=alphas)
    v_width = np.array(v_width)
    y_hat_test = test.y_hat[None, :, :] + margin[:, None, :]

    width_raw = np.array(jax.vmap(
        partial(calibrate.width, y_true=test.y_true))(y_hat=y_hat_test))
    coverage_raw = np.array(jax.vmap(
        partial(calibrate.coverage, y_true=test.y_true))(y_hat=y_hat_test))

    v_best = np.argmin(v_width, axis=1)
    width_selected = width_raw[np.arange(v_best.shape[0]), v_best]
    coverage_selected = coverage_raw[np.arange(v_best.shape[0]), v_best]

    mape_all = loss.PercentError(log=log)(test.y_true, test.y_hat[:, :1])[:, 0]
    mape = np.array(jnp.mean(mape_all))

    return {
        "mape": mape, "val_width": v_width, "val_best": v_best,
        "width": width_selected, "coverage": coverage_selected,
        "margin": np.array(margin), "width_raw": width_raw,
        "coverage_raw": coverage_raw
    }


def _main(args):
    if args.out is None:
        args.out = args.path.replace("results/", "summary/") + ".npz"

    with open(os.path.join(args.path, "config.json")) as f:
        cfg = json.load(f)

    objectives = ObjectiveSet.from_config(cfg["objectives"])
    model = models[cfg["model"]].from_config(objectives, **cfg["model_args"])
    q_test = np.arange(90, 100)
    alphas = (100 - q_test) / 100

    def _evaluate(path):
        splits = Split.from_npz(
            os.path.join(args.splits, path), objectives.objectives)
        checkpoint = os.path.join(args.path, path.replace(".npz", ".pkl"))
        with open(checkpoint, 'rb') as f:
            params = pickle.load(f).params

        ival = {k: v.val for k, v in splits.items()}
        val = model.evaluate(params, model.objectives.index(ival))
        itest = {k: v.test for k, v in splits.items()}
        test = model.evaluate(params, model.objectives.index(itest))
        return {
            k: _metrics(
                val[k], test[k], alphas, log=cfg["objectives"]["mf"]["log"])
            for k in val}

    def _evaluate2(split):
        replicates = sorted(os.listdir(os.path.join(args.splits, split)))
        return utils.tree_stack(
            [_evaluate(os.path.join(split, r)) for r in replicates], _np=np)

    split_levels = sorted(os.listdir(args.splits))
    splits_arr = np.array([float(x) for x in split_levels])
    results = utils.tree_stack(
        [_evaluate2(s) for s in tqdm(split_levels)], _np=np)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out, quantiles=q_test, splits=splits_arr,
        **utils.dict_flatten(results))
