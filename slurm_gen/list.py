"""Functionality for listing datasets currently in the cache.

This module provides a CLI, which is probably the easiest way to use it.

Kyle Roth. 2019-10-30.
"""


import argparse
import os
import sys

from slurm_gen.data_objects import Cache
from slurm_gen.utils import gitHubIssueHandler


def print_sizes(param_set):
    """Print (nicely) the ParamSet and its sizes.

    Parameter strings are often very long, so we print them (and their counts) nicely.

    Args:
        param_set (ParamSet)
    """
    # collect strings of parameters two wide
    param_iter = iter(param_set.name.split("|"))
    param_strings = []
    while True:
        try:
            param_strings.append(next(param_iter) + "|")
            param_strings[-1] += next(param_iter) + "|"
        except StopIteration:
            break

    # collect strings of counts
    count_strings = [" raw: {}".format(param_set.raw_size)]
    for group in param_set:
        count_strings.append(" {}: unprocessed({})".format(group.name, group.unprocessed_size))
        for preproc_set in group:
            count_strings.append(
                " " * (len(group.name) + 1) + ": {}({})".format(preproc_set.name, preproc_set.size)
            )

    # print the param and count strings next to each other
    param_iter = iter(param_strings)
    count_iter = iter(count_strings)

    # right align the param strings, with 4 spaces before the whole thing
    param_spacing = 4 + max(len(string) for string in param_strings)

    while True:
        # print the next strings
        try:
            print(next(param_iter).rjust(param_spacing), end="")
        except StopIteration:
            # print the rest of the count strings, if any
            try:
                while True:
                    print(" " * (param_spacing - 1) + "|" + next(count_iter))
            except StopIteration:
                break
        try:
            print(next(count_iter))
        except StopIteration:
            # print the rest of the param strings, if any
            print()
            try:
                while True:
                    print(next(param_iter).rjust(param_spacing))
            except StopIteration:
                break


def single_list(dataset):
    """Print the count information for a single dataset.

    Args:
        dataset (Dataset)
    """
    count = 0
    for param_set in dataset:
        print("Param set #{}:".format(count))
        print_sizes(param_set)
        count += 1


def _list(dataset=None, verbose=False):
    """List the datasets along with the number of samples generated for them.

    Args:
        dataset (str): if given, list only the number of samples for this dataset.
        verbose (bool): whether to print debug statements.
    """
    if dataset is None:
        # list all of them!
        divider = "-" * 80 + "\n"
        did_print = False

        for d_set in Cache(os.getcwd(), verbose):
            if did_print:
                print(divider)
            print("{}:".format(d_set.name))
            single_list(d_set)
            did_print = True

        if not did_print:
            print("No datasets found. Generate one!")
    else:
        # get list the one dataset
        single_list(Cache(os.getcwd(), verbose)[dataset])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="list samples in the cache for the current directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset", nargs="?", help="dataset for which to list quantities")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print debug info while running the command"
    )

    args = parser.parse_args()

    gitHubIssueHandler(_list, args.dataset, args.verbose)
