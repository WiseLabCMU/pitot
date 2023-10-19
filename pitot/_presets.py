"""Configuration presets."""


# Quantile regression
def _quantile(quantiles):
    return {
        ("model_args", "loss_class"): "Pinball",
        ("model_args", "loss_args"): {"quantiles": quantiles}}


# Interference weight
def _if_weight(w):
    return {
        ("objectives", "if2", "weight"): w,
        ("objectives", "if3", "weight"): w,
        ("objectives", "if4", "weight"): w
    }


# Use percent error loss instead of log loss
_NOLOG = {
    ("objectives", "mf", "log"): False,
    ("objectives", "if2", "log"): False,
    ("objectives", "if3", "log"): False,
    ("objectives", "if4", "log"): False,
    ("model_args", "loss_class"): "PercentError",
    ("model_args", "loss_args"): {"log": False},
    ("model_args", "do_baseline"): False
}

# Disable platform features
_NOPLATFORM = {
    ("model_args", "platform_embedding"): "MFEmbedding",
    ("model_args", "platform_args"): {},
}

# Disable workload features
_NOWORKLOAD = {
    ("model_args", "workload_embedding"): "MFEmbedding",
    ("model_args", "workload_args"): {}
}

# No interference modeling
_NOINTERFERENCE = {
    ("model",): "PitotIgnore",
    ("model_args",): {
        "platform_embedding": "HybridEmbedding",
        "platform_args": {"learned_features": 4, "layers": [128, 128]},
        "workload_embedding": "HybridEmbedding",
        "workload_args": {"learned_features": 4, "layers": [128, 128]},
        "loss_class": "Squared",
        "loss_args": {},
        "optimizer": "adamaxw",
        "optimizer_args": {"learning_rate": 0.001, "b1": 0.9, "b2": 0.999},
        "mf_dim": 32,
        "do_baseline": True
    }
}


# Set learned features
def _learned_features(d):
    return {
        ("model_args", "platform_args", "learned_features"): d,
        ("model_args", "workload_args", "learned_features"): d
    }


PRESETS = {
    "pitot": {},
    # -- Ablations: embedding dimension ---------------------------------------
    "embedding/4": {("model_args", "mf_dim"): 4},
    "embedding/8": {("model_args", "mf_dim"): 8},
    "embedding/16": {("model_args", "mf_dim"): 16},
    # embedding/32 == pitot
    "embedding/64": {("model_args", "mf_dim"): 64},
    "embedding/128": {("model_args", "mf_dim"): 128},
    # -- Ablations: learned features ------------------------------------------
    "features/0": _learned_features(0),
    "features/1": _learned_features(1),
    "features/2": _learned_features(2),
    # features/4 == pitot
    "features/8": _learned_features(8),
    "features/16": _learned_features(16),
    # -- Ablations: interference types ----------------------------------------
    "interference/1": {("model_args", "if_dim"): 1},
    # interference/2 == pitot
    "interference/4": {("model_args", "if_dim"): 4},
    "interference/8": {("model_args", "if_dim"): 8},
    "interference/16": {("model_args", "if_dim"): 16},
    # -- Ablations: objective weight ------------------------------------------
    "weight/0.1": _if_weight(0.1 / 3),
    "weight/0.2": _if_weight(0.2 / 3),
    "weight/0.5": _if_weight(0.5 / 3),
    "weight/1.0": _if_weight(1.0 / 3),
    "weight/2.0": _if_weight(2.0 / 3),
    # -- Ablations: side information ------------------------------------------
    "features/noworkload": _NOWORKLOAD,
    "features/noplatform": _NOPLATFORM,
    "features/blackbox": {**_NOPLATFORM, **_NOWORKLOAD},
    # -- Ablations: design components -----------------------------------------
    # components/full == pitot
    "components/nobaseline": {("model_args", "do_baseline"): False},
    "components/naiveloss": _NOLOG,
    "components/discard": {**_NOINTERFERENCE, **_if_weight(0.0)},
    "components/ignore": _NOINTERFERENCE,
    # -- Ablations: conformal regression --------------------------------------
    # conformal/nonquantile == pitot
    "conformal/naive": _quantile([90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
    "conformal/optimal": _quantile([50, 60, 70, 80, 90, 95, 98, 99]),
    # -- Baselines ------------------------------------------------------------
    "baseline/monolith": {
        ("model",): "Monolith",
        ("model_args",): {
            "layers": [256, 256],
            "loss_class": "Squared",
            "loss_args": {},
            "optimizer": "adamaxw",
            "optimizer_args": {"learning_rate": 0.001, "b1": 0.9, "b2": 0.999}}
    },
    "baseline/attention": {
        ("model",): "Attention",
        ("model_args",): {
            "embedding_layers": [256, 256], "output_layers": [32],
            "attention_dim": 32, "value_dim": 32}
    }
}
