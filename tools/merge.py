"""Merge matrix and side information."""

import numpy as np

_desc = "Merge execution time matrix, module/runtime side information."


def _parse(p):
    p.add_argument("-m", "--matrix", default="matrix.npz")
    p.add_argument("-p", "--opcodes", default="opcodes.npz")
    p.add_argument("-t", "--platforms", default="platforms.npz")
    p.add_argument("-o", "--out", default="data.npz")
    p.add_argument(
        "-d", "--plot", help="Draw plot.", action='store_true', default=False)
    return p


def _main(args):
    matrix = np.load(args.matrix)
    opcodes = np.load(args.opcodes)
    platform = np.load(args.platforms)

    assert np.all(matrix["platform"] == platform["platform"])
    assert np.all(matrix["module"] == opcodes["module"])
    assert platform["data"].shape[1] == platform["feature"].shape[0]
    assert opcodes["data"].shape[1] == opcodes["opcode"].shape[0]

    valid_modules = np.sum(matrix["data"] > 0, axis=0) > 1
    valid_platforms = np.sum(matrix["data"] > 0, axis=1) > 1

    np.savez(
        args.out, data=matrix["data"][valid_platforms, :][:, valid_modules],
        module=matrix["module"][valid_modules],
        module_data=opcodes["data"][valid_modules],
        module_features=opcodes["opcode"],
        platform=matrix["platform"][valid_platforms],
        platform_data=platform["data"][valid_platforms],
        platform_features=platform["feature"])
