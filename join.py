"""Join datasets by runtime."""

import numpy as np
import argparse


def _load_pair(ds, op):
    ds = np.load(ds, allow_pickle=True)
    op = np.load(op, allow_pickle=True)

    np.testing.assert_array_equal(
        ds['files'], [f.replace('wasm', 'aot') for f in op['files']])

    return ds, op


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Join datasets by runtime.")
    parser.add_argument(
        "--datasets", nargs='+', default=[], help="Dataset file paths to join.")
    parser.add_argument(
        "--wasm", nargs='+', default=[], help="WASM opcode counts to join.")
    parser.add_argument("--out", default="merged.npz", help="Output npz file.")
    args = parser.parse_args()

    # Load, check files match with WASM
    npzs = [_load_pair(ds, op) for ds, op in zip(args.datasets, args.wasm)]

    # Check runtimes match
    for ds, _ in npzs[1:]:
        np.testing.assert_array_equal(npzs[0][0]['runtimes'], ds['runtimes'])

    # Merge
    meta = {}
    meta['files'] = np.concatenate([z['files'] for z, _ in npzs]).astype(str)
    meta['runtimes'] = np.array(npzs[0][0]['runtimes']).astype(str)
    meta['opcodes'] = np.concatenate([z['opcodes'] for _, z in npzs])
    data = {}
    for key in npzs[0][0].keys():
        if key not in {'files', 'runtimes'}:
            data[key] = np.concatenate([z[key] for z, _ in npzs], axis=-2)

    # Save
    np.savez(args.out, **meta, **data)
