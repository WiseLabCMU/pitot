"""Prediction management utility dispatcher."""

import printtools as pt
from argparse import ArgumentParser, RawTextHelpFormatter

from forecast import scripts


ACTIONS = {
    "plot_grid": (
        scripts.plot_grid, "Plot trace or histogram from benchmarks."),
    "matrix": (scripts.matrix, "Generate execution time matrix."),
    "runtimes": (scripts.runtimes, "Create runtimes.npz metadata file."),
    "opcodes": (scripts.opcodes, "Create opcodes.npz metadata file."),
    "join": (scripts.join, "Join matrices and metadata into dataset."),
    "if_table": (
        scripts.interference, "Generate interference summary table."),
    "if_plot": (
        scripts.plot_interference,
        "Plot interference marginal distributions by runtime."),
    "interference": (
        [scripts.interference, scripts.plot_interference],
        "Create interference summary and plot.")
}


def _help_table():
    return pt.table(
        [[k, v[1]] for k, v in ACTIONS.items()], vline=False, render=True)


def _parser(mod, args):
    parser = mod._parse()
    parser.add_argument(
        "script", help="Script: {}".format(args.script), nargs='?')
    return parser


def _parse_main():
    p = ArgumentParser(
        add_help=False, description="Cluster management scripts. Use "
        "python3 manage.py --config <path/to/config.json> <script> <args> ... "
        "to run.", formatter_class=RawTextHelpFormatter)
    p.add_argument(
        "script", default=None, nargs='?',
        help="Script/action to run (* password required)\n" + _help_table())
    p.add_argument(
        "-h", "--help", action="store_true",
        help="show this help message if first; otherwise, passes through.")
    p.add_argument("--config", help=(
        "Config file to load; priority is (1) explicitly passed args, "
        "(2) config file, (3) defaults"))
    p.set_defaults(help=False)
    return p


if __name__ == '__main__':

    p = _parse_main()
    args, _ = p.parse_known_args()

    if args.help:
        if args.script is None:
            p.print_help()
            exit(0)

    if args.script in ACTIONS:
        script = ACTIONS[args.script][0]
        if not isinstance(script, list):
            script = [script]

        for s in script:
            s._main(_parser(s, args).parse_args())
    else:
        print("Invalid script: {}\n".format(args.script))
        print("Options:")
        print(_help_table())
