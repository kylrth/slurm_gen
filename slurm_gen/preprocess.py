"""Command line interface for SLURM_gen to preprocess samples.

Kyle Roth. 2019-11-06.
"""


import argparse
import os

from slurm_gen import utils
from slurm_gen.data_objects import Cache


def _preprocess(dataset, preprocessor, target, param_set, size, yes, verbose):
    """Preprocess samples in a set of data using the specified preprocessor.

    Args:
        dataset (class): dataset defined in datasets.py in the current directory.
        preprocessor (class): a preprocessor for the dataset; also defined in datasets.py.
        target (str): name of target group, e.g. "train" or "test".
        param_set (utils.DefaultParamObject): identifier of the chosen parameter set.
        size (int): final number of samples desired in the preprocessed set.
        yes (bool): if True, confirmation won't be asked.
        verbose (bool): whether to print debug statements.
    """
    # make sure the param set has enough samples
    param_set = Cache(os.getcwd(), verbose=verbose)[dataset][param_set]
    group = param_set[target]
    if param_set.raw_size + group.unprocessed_size < size:
        print("This param set only has {} samples. Aborting".format(param_set.raw_size))
        return

    print("Dataset:", dataset.__name__)
    print("Params:", param_set.name)
    print("Group:", target)

    # make sure the group has enough unprocessed samples
    if group.unprocessed_size < size:
        print("This group only has {} samples.".format(group.unprocessed_size))
        if yes:
            print("Moving {} samples into the group".format(size - group.unprocessed_size))
        else:
            # ask confirmation
            print(
                "Would you like to move {} samples into this group?".format(
                    size - group.unprocessed_size
                ),
                end="",
            )
            confirm = input(" (y/N): ")
            if confirm.lower() not in {"y", "yes"}:
                print("Aborting")
                return

        param_set.move(target, size - group.unprocessed_size)

    # ask confirmation
    try:
        current_size = group[preprocessor].size
    except IndexError:
        current_size = 0
    if yes:
        print(
            "Preprocessing {} samples with '{}'".format(size - current_size, preprocessor.__name__)
        )
        print("for a total of {} samples preprocessed this way.".format(size))
    else:
        print(
            "Would you like to preprocess {} samples with '{}'?".format(
                size - current_size, preprocessor.__name__
            )
        )
        print(
            "This would result in a total of {} samples preprocessed this way.".format(size), end=""
        )
        confirm = input(" (y/N): ")
        if confirm.lower() not in {"y", "yes"}:
            print("Aborting")
            return

    group.assert_preprocessed_size(preprocessor, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command line interface for preprocessing samples in SLURM_gen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required arguments
    parser.add_argument("dataset", help="dataset containing samples to preprocess")
    parser.add_argument(
        "preprocessor",
        help="preprocessor function to use. Must be defined in datasets.py as a preprocessor for "
        + "`dataset`",
    )
    parser.add_argument(
        "target", help="set containing samples to be preprocessed (e.g. 'train' or 'val')"
    )
    parser.add_argument(
        "n_samples", type=int, help="desired minimum size of preprocessed sample set"
    )
    parser.add_argument(
        "-p",
        "--param_set",
        required=True,
        type=utils.paramSetIdentifier,
        help="param set identifier. May be the number given by the `list` command, a dictionary of "
        "parameters and values, or the string produced by a ParamObject.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print debug info while running the command"
    )

    args = parser.parse_args()

    # get the actual preprocessing function
    args.dataset = utils.get_dataset(args.dataset)
    args.preprocessor = utils.get_preprocessor(args.preprocessor, args.dataset)

    try:
        _preprocess(
            args.dataset,
            args.preprocessor,
            args.target,
            args.param_set,
            args.n_samples,
            args.yes,
            args.verbose,
        )
    except KeyboardInterrupt:
        print("\nexiting")
