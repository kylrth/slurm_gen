"""Command line interface for SLURM_gen.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen import utils
from slurm_gen.data_objects import Cache


def _move(dataset, param_num, size, target):
    """Move raw samples to a group, confirming the choice before doing so.

    Args:
        dataset (str)
        param_num (int): ordered number corresponding to the chosen parameter set.
        size (int): number of samples to move.
        target (str): name of target group, e.g. "train" or "test".
    """
    if target == "train+val":
        raise ValueError(
            'the name "train+val" is reserved for combining the "train" and "val" sets'
        )

    param_set = Cache()[dataset][param_num]

    if param_set.raw_size < size:
        print("This param set only has {} samples. Aborting".format(param_set.raw_size))
        return

    # ask confirmation
    print("Dataset:", dataset)
    print("Params:", param_set.name)
    print("Move {} raw samples to {}?".format(size, target))
    target_size = param_set[target].unprocessed_size if target in param_set else 0
    print(
        "This would result in a total of {} samples in '{}',".format(
            target_size + size, target
        )
    )
    print("with {} raw samples left over.".format(param_set.raw_size - size), end="")
    confirm = input(" (y/N): ")
    if confirm.lower() not in {"y", "yes"}:
        print("Aborting")
        return

    param_set.move(target, size)
    os.remove(os.path.join(param_set.path, "raw", ".metadata"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line interface for SLURM_gen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required arguments
    parser.add_argument("dataset", help="Dataset containing samples to move.")
    parser.add_argument(
        "-p",
        "--param_num",
        required=True,
        type=int,
        help="Param set number given by the list command.",
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Raw samples will be moved to this location.",
    )
    parser.add_argument(
        "-n", "--n_samples", required=True, type=int, help="Number of samples to move."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print debug info while running the command.",
    )

    args = parser.parse_args()
    if args.param_num is None:
        raise ValueError('--param_num must be set for the "move" command')
    if args.target is None:
        raise ValueError('--target must be set for the "move" command')
    if args.n_samples is None:
        raise ValueError('--n_samples must be set for the "move" command')

    try:
        _move(args.dataset, args.param_num, args.n_samples, args.target)
    except KeyboardInterrupt:
        print("\nexiting")
