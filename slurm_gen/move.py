"""Command line interface for SLURM_gen to move samples into sets.

Kyle Roth. 2019-07-27.
"""


import argparse
import ast
import os
import sys

from slurm_gen import utils
from slurm_gen.data_objects import Cache


def _move(dataset, param_num, size, target, verbose):
    """Move raw samples to a group, confirming the choice before doing so.

    Args:
        dataset (str)
        param_set: identifier of the chosen parameter set, produced by paramSetIdentifier.
        size (int): number of samples to move.
        target (str): name of target group, e.g. "train" or "test".
        verbose (bool): whether to print debug statements.
    """
    if target == "train+val":
        raise ValueError(
            'the name "train+val" is reserved for combining the "train" and "val" sets'
        )

    param_set = Cache(os.getcwd(), verbose)[dataset][param_num]

    if param_set.raw_size < size:
        print("This param set only has {} samples. Aborting".format(param_set.raw_size))
        return

    # ask confirmation
    print("Dataset:", dataset)
    print("Params:", param_set.name)
    print("Move {} raw samples to {}?".format(size, target))
    target_size = param_set[target].unprocessed_size if target in param_set else 0
    print("This would result in a total of {} samples in '{}',".format(target_size + size, target))
    print("with {} raw samples left over.".format(param_set.raw_size - size), end="")
    confirm = input(" (y/N): ")
    if confirm.lower() not in {"y", "yes"}:
        print("Aborting")
        return

    param_set.move(target, size)


def paramSetIdentifier(arg):
    """Convert the argument string to a unique identifier of a parameter set.

    Possible arguments are:
    - a whole number identifier (0, 1, 2, ...)
    - a string that ast.literal_eval can convert to a dictionary
      (e.g. "{'left': 0, 'std_dev': 0.5}")
    - a string created by a ParamObject (e.g. "left#0|right#1|std_dev#0.5")
    """
    # whole number identifier
    try:
        return int(arg)
    except ValueError:
        pass

    # dictionary
    try:
        return ast.literal_eval(arg)
    except ValueError:
        pass

    # ParamObject string
    generator = utils.get_generator(sys.argv[1])
    return generator.paramClass._from_string(arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="move samples into specific groups",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset", help="dataset containing samples to move")
    parser.add_argument("n_samples", type=int, help="number of samples to move")
    parser.add_argument("target", help="raw samples will be moved to this location")
    parser.add_argument(
        "-p",
        "--param_set",
        required=True,
        type=paramSetIdentifier,
        help="param set identifier. May be the number given by the `list` command, a dictionary of "
        "parameters and values, or the string produced by a ParamObject.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print debug info while running the command"
    )

    args = parser.parse_args()

    try:
        _move(args.dataset, args.param_set, args.n_samples, args.target, args.verbose)
    except KeyboardInterrupt:
        print("\nexiting")
