"""Script dispatcher."""

from argparse import ArgumentParser
import importlib


def dispatch(target: str) -> None:
    """Dispatch scripts to the specified module.
    
    The module should have a `_scripts` attribute with the list of valid
    scripts; the `__doc__` is used as the script description.
    """
    target_module = importlib.import_module(target)
    commands = {
        cmd: importlib.import_module("{}.{}".format(target, cmd))
        for cmd in target_module._scripts
    }
    parser = ArgumentParser(description=target_module.__doc__)

    subparsers = parser.add_subparsers()
    for name, command in commands.items():
        p = subparsers.add_parser(
            name, help=command.__doc__, description=command.__doc__)
        command._parse(p)
        p.set_defaults(_func=command._main)

    args = parser.parse_args()
    args._func(args)


if __name__ == '__main__':
    dispatch("scripts")
