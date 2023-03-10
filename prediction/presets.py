"""Configuration presets."""

import copy


DEFAULT = {
    "model": "embedding",
    "model_args": {
        "X_p": True, "X_m": True, "alpha": 0.002,
        "layers": [128, 128], "dim": 8, "scale": 0.01},
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
    # Linear
    "linear/32": _linear(32),
    "linear/64": _linear(64),
    "linear/128": _linear(128),
    "linear/256": _linear(256),
    "linear/512": _linear(512)
}


def _override(default: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        _subdict = default
        for subkey in k[:-1]:
            _subdict = _subdict[subkey]
        _subdict[k[-1]] = v
    return default


def get_config(config: str) -> dict:
    """Get configuration preset by name."""
    return _override(copy.deepcopy(DEFAULT), PRESETS[config])


def match(config: str) -> list[str]:
    """Get configuration presets matching the given name."""
    return [k for k in PRESETS if k.startswith(config.rstrip('*'))]
