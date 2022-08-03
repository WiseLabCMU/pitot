"""Format runtime manifests."""

import argparse
import numpy as np
import json


def _int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0


def _logsize(x, offset=30):
    """Log base 2 of size."""
    try:
        return np.log2(int(x)) - offset
    except (ValueError, TypeError):
        return 0


def _loghz(x):
    try:
        return np.log(int(x) / 10**6)
    except (ValueError, TypeError):
        return 0


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir", default="data/runtimes.json", help="Dataset directory.")
    p.add_argument(
        "--out", default="data/runtimes.npz", help="Output file.")
    return p


def _main(args):
    with open(args.dir) as f:
        data = json.load(f)

    architectures = np.unique(
        [v['platform']['cpu']['class'] for _, v in data.items()])

    def _one_hot(x):
        return (architectures == x).astype(np.float32)

    def _transform(m):
        return [
            # m['cpu']['arch'],
            *_one_hot(m['cpu']['class']),     # Architecture
            # m['cpu']['cores'],
            _loghz(m['cpu']['cpufreq']),     # Log CPU Frequency / 1GHz
            _logsize(m['mem']['l1d_size']),  # Log L1D
            _logsize(m['mem']['l1i_size']),  # Log L1I
            _logsize(m['mem']['l2_size']),   # Log L2
            _logsize(m['mem']['l2_line']),       # Log L2 line size
            _int(m['mem']['l2_assoc']),      # L2 associativity
            _logsize(m['mem']['l3_size']),   # L3 size in MiB
            # logsize(m['mem']['total'], 30)
        ]

    np.savez(
        args.out,
        data=np.array([_transform(v['platform']) for _, v in data.items()]),
        runtimes=np.array([v['name'] for _, v in data.items()]),
        cpus=np.array(architectures)
    )


if __name__ == '__main__':
    _main(_parse().parse_args())
