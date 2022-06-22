"""Generate data plots."""

from dataset import Session
from parse import ArgumentParser


def _parse():
    p = ArgumentParser(description="Generate plots from benchmarking data.")
    p.add_argument(
        "path", nargs='+', default=['data'],
        help="Directories to plot data from; must be in orchestrator format.")
    p.add_argument(
        "--keys", nargs='+', default=['cpu_time'],
        help="Keys to plot, i.e. cpu_time, wall_time, etc.")
    p.add_to_parser("plot", Session.plot_grid, group="plot", exclude='keys')
    return p


if __name__ == '__main__':
    args = _parse().parse_args()

    dataset = Session(args["path"])
    dataset.plot_grid(keys=args["keys"], **args["plot"])
