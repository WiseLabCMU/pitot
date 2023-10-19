"""Configuration defaults and preset management."""

import copy

from ._presets import PRESETS


_OBJECTIVES = {
    "mf": {
        "path": ["data/data.npz"],
        "axes": {"platform": "platform", "workload": "workload"},
        "batch": 512, "log": True, "weight": 1.0},
    "if2": {
        "path": ["data/data.npz", "data/if2.npz"],
        "axes": {
            "platform": "platform", "workload": "workload",
            "interference0": "workload"},
        "batch": 512, "log": True, "weight": 0.5 / 3},
    "if3": {
        "path": ["data/data.npz", "data/if3.npz"],
        "axes": {
            "platform": "platform", "workload": "workload",
            "interference0": "workload", "interference1": "workload"},
        "batch": 512, "log": True, "weight": 0.5 / 3},
    "if4": {
        "path": ["data/data.npz", "data/if4.npz"],
        "axes": {
            "platform": "platform", "workload": "workload",
            "interference0": "workload", "interference1": "workload",
            "interference2": "workload"},
        "batch": 512, "log": True, "weight": 0.5 / 3}
}

_DEFAULT_MODEL = {
    "platform_embedding": "HybridEmbedding",
    "platform_args": {"learned_features": 1, "layers": [128, 128]},
    "workload_embedding": "HybridEmbedding",
    "workload_args": {"learned_features": 1, "layers": [128, 128]},
    "loss_class": "Squared",
    "loss_args": {},
    "optimizer": "adamaxw",
    "optimizer_args": {"learning_rate": 0.001, "b1": 0.9, "b2": 0.999},
    "mf_dim": 32,
    "if_dim": 2,
    "if_slope": 0.1,
    "do_baseline": True
}

DEFAULT = {
    "objectives": _OBJECTIVES,
    "model": "Pitot",
    "model_args": _DEFAULT_MODEL,
    "training_args": {"steps": 20000, "val_every": 200},
    "seed": 42
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


def check_equal(x, y) -> bool:
    """Check if two (config) objects are equal."""
    if isinstance(x, list):
        if not isinstance(y, list):
            return False
        for xi, yi in zip(x, y):
            if not check_equal(xi, yi):
                return False
        return True
    elif isinstance(x, dict):
        if not isinstance(y, dict):
            return False
        for kx, ky in zip(x, y):
            if not check_equal(x[kx], y[ky]):
                return False
        return True
    else:
        return x == y


def assert_equal(x, y, xlabel="Config A", ylabel="Config B") -> None:
    """Check if two (config) objects are equal (with consequences)."""
    if not check_equal(x, y):
        raise ValueError(
            "Configurations do not match!\n\n{}: {}\n{}: {}".format(
                xlabel, x, ylabel, y))


def get_config(config: str) -> dict:
    """Get configuration preset by name."""
    return _override(copy.deepcopy(DEFAULT), PRESETS[config])


def match(config: str) -> list[str]:
    """Get configuration presets matching the given name."""
    return [k for k in PRESETS if k.startswith(config)]
