"""Configuration presets."""

DEFAULT = {
    "model": "embedding",
    "model_args": {
        "X_m": True, "X_d": True, "alpha": 0.001,
        "layers": [64, 64], "dim": 4, "scale": 0.01},
    "training_args": {
        "beta": (1.0, 0.0), "batch": (128, 0), "replicates": 50,
        "k": 10, "do_baseline": True, "if_adjust": 2800
    }
}


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
    "Er8": {("model_args", "layers"): [64, 8]},
    "Er16": {("model_args", "layers"): [64, 16]},
    "Er32": {("model_args", "layers"): [64, 32]},
    "Er64": {("model_args", "layers"): [64, 64]},
    "Er128": {("model_args", "layers"): [64, 128]},
    # Ablations - Embedding Dimension, Linear (r=64)
    "Lr1": _cfg_linear(1),
    "Lr2": _cfg_linear(2),
    "Lr4": _cfg_linear(4),
    "Lr8": _cfg_linear(8),
    "Lr16": _cfg_linear(16),
    "Lr32": _cfg_linear(32),
    "Lr64": _cfg_linear(64),
    "Lr128": _cfg_linear(128),
    # Ablations - Embedding Input Dimension (q=4)
    "Eq2": {("model_args", "dim"): 2},
    "Eq4": {("model_args", "dim"): 4},
    "Eq8": {("model_args", "dim"): 8},
    "Eq16": {("model_args", "dim"): 16},
    "Eq32": {("model_args", "dim"): 32},
    # Ablations - Interference Types (s=?)
    "Is1": _cfg_if(1),
    "Is2": _cfg_if(2),
    "Is3": _cfg_if(3),
    "Is4": _cfg_if(4),
    # Full experiments
    # for linear use linear = Lr64
    # for interference use interference = Is1
    # for embedding use embedding = Eq4 = Er64
    "ignore": {
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    },
    "device_only": {("model_args", "X_m"): None},
    "module_only": {("model_args", "X_d"): None},
    # Other datasets: interference
    "if3": _cfg_if(1),
    "if3.mc": _cfg_if(1),
    "if.mc": _cfg_if(1),
    # Other datasets: embedding (discard)
    "e.if3": {},
    "e.if3.mc": {},
    "e.if.mc": {},
    # Other datasets: ignore
    "ig.if3.mc": {
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    },
    "ig.if3": {
        ("training_args", "beta"): (1.0, 1.0),
        ("training_args", "batch"): (128, 128)
    },
    # Baselines
    "naive_mlp": {
        ("model",): "naive_mlp",
        ("model_args",): {"layers": (64, 64), "alpha": 0.1}
    },
    "device_mlp": {
        ("model",): "device_mlp",
        ("model_args",): {"layers": (32,), "alpha": 0.1}
    },
    # Ablations - no baseline
    "embedding_nb": {("training_args", "do_baseline"): False},
    "naive_mlp": {("training_args", "do_baseline"): False},
    "device_mlp": {("training_args", "do_baseline"): False},
    "linear_nb": {("training_args", "do_baseline"): False, **_cfg_linear(64)}
}
