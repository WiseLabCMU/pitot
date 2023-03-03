"""Generate data plots."""

from forecast.dataset import Session
from libsilverline import ArgumentParser


def _parse():
    p = ArgumentParser(description="Generate plot grid from benchmarks.")
    p.add_argument(
        "--path", "-p", nargs='+', default=['data'],
        help="Directories to plot data from; must be in orchestrator format.")
    p.add_argument(
        "--key", "-k", nargs='+', default=['cpu_time'],
        help="Keys to plot, i.e. cpu_time, wall_time, etc.")
    p.add_to_parser("plot", Session.plot_grid, group="plot", exclude='keys')
    p.add_argument(
        "--out", "-o", default=None, help="Path to save plot to."
        "If blank, will save to `{path[0]}.{mode:trace,hist}.png`.")
    return p


def _main(args):

    dataset = Session(args["path"])
    fig, _ = dataset.plot_grid(keys=args["key"], **args["plot"])

    if args["out"]:
        fig.savefig(args["out"])
    else:
        fig.savefig("{}.{}.png".format(args["path"][0], args["plot"]["mode"]))
