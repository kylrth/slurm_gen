"""Command line interface for SLURM_gen to move samples into sets.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen.data_objects import Cache


def _move(dataset, param_num, size, target, verbose):
    """Move raw samples to a group, confirming the choice before doing so.

    Args:
        dataset (str)
        param_num (int): ordered number corresponding to the chosen parameter set.
        size (int): number of samples to move.
        target (str): name of target group, e.g. "train" or "test".
        verbose (bool): whether to print debug statements.
    """
    if target == "train+val":
        raise ValueError(
            'the name "train+val" is reserved for combining the "train" and "val" sets'
        )

    param_set = Cache(verbose=verbose)[dataset][param_num]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command line interface for moving samples in SLURM_gen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset", help="dataset containing samples to move")
    parser.add_argument(
        "-p",
        "--param_num",
        required=True,
        type=int,
        help="param set number given by the `list` command",
    )
    parser.add_argument(
        "-t", "--target", required=True, help="raw samples will be moved to this location"
    )
    parser.add_argument(
        "-n", "--n_samples", required=True, type=int, help="number of samples to move"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print debug info while running the command"
    )

    args = parser.parse_args()

    try:
        _move(args.dataset, args.param_num, args.n_samples, args.target, args.verbose)
    except KeyboardInterrupt:
        print("\nexiting")
