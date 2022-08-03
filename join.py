"""Join datasets by runtime."""

import numpy as np
import argparse


def _parse():
    parser = argparse.ArgumentParser(description="Join datasets by runtime.")
    parser.add_argument(
        "--datasets", nargs='+', default=[], help="Dataset paths to join.")
    parser.add_argument(
        "--opcodes", nargs='+', default=[], help="WASM opcode counts to join.")
    parser.add_argument(
        "--runtimes", nargs='+', default=[], help="Runtime metadata to join.")
    parser.add_argument("--out", default="merged.npz", help="Output npz file.")
    return parser


def _load_matrix(sources):
    # Load, check files match with WASM
    npzs = [np.load(ds) for ds in sources]

    # Check runtimes match
    runtimes = npzs[0]['runtimes']
    for ds in npzs[1:]:
        np.testing.assert_array_equal(runtimes, ds['runtimes'])

    # Concatenate
    data = {}
    for key in npzs[0].keys():
        if key not in {'files', 'runtimes'}:
            data[key] = np.concatenate([z[key] for z in npzs], axis=-2)

    # Modules
    modules = np.concatenate([z['files'] for z in npzs]).astype(str)

    return data, runtimes, modules


def _load_side_info(sources, keys, values):
    res = {}
    for source in sources:
        npz = np.load(source)
        for k, v in zip(npz[keys], npz[values]):
            res[k] = v
    return res


if __name__ == '__main__':
    args = _parse().parse_args()

    data, runtimes, modules = _load_matrix(args.datasets)
    opcodes = _load_side_info(args.opcodes, "files", "opcodes")
    opcodes = {k.replace('wasm', 'aot'): v for k, v in opcodes.items()}
    platform = _load_side_info(args.runtimes, "runtimes", "data")

    # Load runtimes
    meta = {
        'modules': modules,
        'runtimes': runtimes,
        'module_data': np.array([opcodes[k] for k in modules]),
        'runtime_data': np.array([platform[k] for k in runtimes])    
    }

    # Save
    np.savez_compressed(args.out, **data, **meta)
