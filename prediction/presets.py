"""Configuration presets."""

import copy


DEFAULT = {
    "model": "embedding",
    "model_args": {
        "X_p": True, "X_m": True, "alpha": 0.002,
        "layers": [128, 128], "dim": 4, "scale": 0.01},
    "objectives": [{
        "batch_size": 2048, "weight": 1.0, "log": True,
        "name": "mf", "save": None, "xkey": "x", "ykey": "y"
    }],
    "training_args": {"replicates": 25, "k": 10, "do_baseline": True},
}


def _linear(dim):
    return {
        ("model",): "linear",
        ("model_args",): {"dim": dim, "alpha": 0.002, "scale": 0.01}
    }


def _paragon(dim):
    return {
        ("model",): "linear",
        ("model_args",): {"dim": dim, "alpha": 0.002, "scale": 0.01},
        ("training_args", "do_baseline"): False,
        ("objectives", 0, "log"): False
    }


def _if_objective(weight):
    return {
        ("objectives", 0, "batch_size"): 128,
        ("objectives", -1): {
            "batch_size": 128, "weight": weight, "log": True, "name": "if",
            "save": "C_ijk_hat", "xkey": "if_x", "ykey": "if_y"
        }
    }


def _if_model(s):
    return {
        ("model",): "interference",
        ("model_args", "s"): s,
        **_if_objective(1.0)
    }


PRESETS = {
    # Embedding Dimension
    "embedding/32": {("model_args", "layers"): [128, 32]},
    "embedding/64": {("model_args", "layers"): [128, 64]},
    "embedding/128": {("model_args", "layers"): [128, 128]},
    "embedding/256": {("model_args", "layers"): [128, 256]},
    "embedding/512": {("model_args", "layers"): [128, 512]},
    # Number of Learned Features
    "features/0": {("model_args", "dim"): 0},
    "features/2": {("model_args", "dim"): 2},
    "features/4": {("model_args", "dim"): 4},
    "features/8": {("model_args", "dim"): 8},
    "features/16": {("model_args", "dim"): 16},
    # Linear Matrix Factorization
    "linear/32": _linear(32),
    "linear/64": _linear(64),
    "linear/128": _linear(128),
    "linear/256": _linear(256),
    "linear/512": _linear(512),
    # Paragon (Naive MF without log or baseline)
    "paragon/32": _paragon(32),
    "paragon/64": _paragon(64),
    "paragon/128": _paragon(128),
    "paragon/256": _paragon(256),
    "paragon/512": _paragon(512),
    # Interference 2-way (run with -d data/data.if.npz)
    "interference/discard": _if_objective(0.0),
    "interference/ignore": _if_objective(1.0),
    "interference/1": _if_model(1),
    "interference/2": _if_model(2),
    "interference/3": _if_model(3),
    "interference/4": _if_model(4),
    # Interference 3-way (run with -d data/data.if3.npz)
    "interference3/discard": _if_objective(0.0),
    "interference3/ignore": _if_objective(1.0),
    "interference3/2": _if_model(2),
    "interference3/no-smt": _if_model(2),
    # Other baselines
    "baseline/platform_only": {("model_args", "X_m"): None},
    "baseline/module_only": {("model_args", "X_p"): None},
    "baseline/mlp": {
        ("model",): "naive_mlp",
        ("model_args",): {"layers": (128, 128), "alpha": 0.1}
    },
    "baseline/device_mlp": {
        ("model",): "device_mlp",
        ("model_args",): {"layers": (64, 64), "alpha": 0.1},
        ("training_args", "k"): 5,
        ("training_args", "replicates"): 5
    },
}


def _override(default: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        node = default
        for key in k[:-1]:
            node = node[key]
        if isinstance(node, list) and k[-1] == -1:
            node.append(v)
        else:
            node[k[-1]] = v
    return default


def get_config(config: str) -> dict:
    """Get configuration preset by name."""
    return _override(copy.deepcopy(DEFAULT), PRESETS[config])


def match(config: str) -> list[str]:
    """Get configuration presets matching the given name."""
    return [k for k in PRESETS if k.startswith(config.rstrip('*'))]
