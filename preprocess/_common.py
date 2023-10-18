"""Workload/platform naming patterns."""

import json
import os
from beartype.typing import Callable, Optional, Any


WORKLOAD_PATTERNS = {
    ".wasm": "",
    "wasm/": "",
    "data/apps/python/": "",
    "apps/wasi-python3.13": "python",
    "apps/python": "python"
}


def platform_name(device: str, runtime: str) -> str:
    """Platform name string representation."""
    return ":".join([runtime, device])


def workload_name(meta: dict) -> str:
    """Workload name string representation."""
    if isinstance(meta, str):
        name = meta
    else:
        name = ":".join([meta["file"]] + meta.get("args", {}).get("argv", []))
    for k, v in WORKLOAD_PATTERNS.items():
        name = name.replace(k, v)
    return name


def apply_recursive(
    path: str, func: Callable[..., Optional[Any]],
    exclude: set[str] = {"runtimes.json"}, load_json: bool = True
) -> list[Any]:
    """Apply function recursively in file system."""
    res = []
    for p in os.listdir(path):
        pp = os.path.join(path, p)
        if os.path.isdir(pp):
            res += apply_recursive(pp, func, exclude=exclude)
        else:
            if p not in exclude:
                try:
                    if load_json:
                        with open(pp) as f:
                            pp = json.load(f)
                    d = func(pp)
                    if d is not None:
                        res.append(d)
                except json.JSONDecodeError:
                    print("Invalid JSON: {}".format(path))
                except Exception as e:
                    print("Couldn't load: {}. {}".format(p, e))
    return res
