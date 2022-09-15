"""Extract opcodes from special opcode recording session."""

import numpy as np
import argparse
from forecast.dataset import Session


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path", default="data/opcodes",
        help="Dataset directory containing interpreted opcode counts.")
    p.add_argument(
        "--out", default="data/opcodes.npz", help="Output file.")
    return p


def _main(args):
    session = Session(args.path)
    opcodes = np.zeros((len(session.files), 256))
    for i, f in enumerate(session.files):
        opcodes[i] = session.get(
            file=f, runtime=session.runtimes[0]).data[-1][16:]
    np.savez_compressed(args.out, files=session.files, opcodes=opcodes)


if __name__ == '__main__':
    _main(_parse().parse_args())
