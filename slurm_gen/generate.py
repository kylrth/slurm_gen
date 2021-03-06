"""Functionality for generating datasets defined in datasets.py.

Use these modules if you encounter a data_loading.InsufficientSamplesError.

This module provides a CLI, which is probably the easiest way to use it.

Kyle Roth. 2019-05-27.
"""


import argparse
import ast
import os
import pickle
import subprocess
import sys
import time

from slurm_gen import datasets, utils
from slurm_gen.data_objects import Cache


def _remove_metadata(dataset, params, verbose=False):
    """Invalidate any record of the number of raw samples for the dataset and param set.

    Args:
        dataset (class): dataset defined in datasets.py.
        params (dict): parameter dict for generating function.
    """
    param_str = params._to_string()
    try:
        os.remove(os.path.join(".slurm_cache", dataset.__name__, param_str, "raw", ".metadata"))
    except FileNotFoundError:
        pass


def _generate_here(dataset, size, params, verbose=False):
    """Using the current process, generate `size` samples of the dataset, using the
    generation function defined in datasets.py.

    Args:
        dataset (class): dataset defined in datasets.py.
        size (int): number of samples to generate.
        params (DefaultParamObject): parameter object to pass to generating function.
    """
    _remove_metadata(dataset, params, verbose)

    # generate the data
    dataset.call(size, params._to_string())


def sanitize_str(s):
    """Sanitize strings to send to a shell.

    Args:
        s (str): string (or something to be converted to a string).
    Returns:
        (str): sanitized string.
    """
    return str(s).replace('"', '\\"').replace("'", "\\'")


# the base string for submitting a data generation job to SLURM
raw_SLURM = """echo '#!/bin/bash

{bash_commands}

cd "{cwd}"
python3 -u -c "from {this}.utils import get_dataset as g; g(\\"{mod}\\").call({size}, \\"{params}\\")"
' | sbatch --error="{out}/%j.out" --output="{out}/%j.out" --time="{time}" """


def create_SLURM_command(dataset, size, params, SLURM_out, job_time):
    """Create the SLURM command that will generate the requested number of samples.

    Args:
        dataset (class): A dataset from datasets.py in the current directory.
        size (int): number of samples to generate.
        params (dict): parameter dict to pass to generating function. Must be reconstructable from
                       ast.literal_eval.
        SLURM_out (str): path to SLURM output directory.
        job_time (str): time for each job to run. Acceptable time formats include "MM", "MM:SS",
                        "HH:MM:SS", "D-HH", "D-HH:MM" and "D-HH:MM:SS". If None, the second
                        standard deviation above the mean of previous runs is used (adapted to the
                        number of samples per job).
    Returns:
        (str): viable bash command to submit SLURM job.
    """
    # create command string
    command = (
        raw_SLURM.format(
            bash_commands="\n".join(dataset.bash_commands),
            cwd=os.getcwd(),
            this=os.path.basename(os.path.dirname(os.path.abspath(__file__))),  # "slurm_gen"
            mod=sanitize_str(dataset.__name__),
            size=sanitize_str(size),
            params=sanitize_str(params._to_string()),
            out=sanitize_str(SLURM_out),
            time=sanitize_str(job_time),
        )
        + dataset.slurm_options
    )

    return command


def _get_sample_time(dataset, params, verbose):
    """Determine the time required to generate a single sample, adjusted two standard deviations
    above the mean.

    Args:
        dataset (class): A dataset from datasets.py.
        params (utils.DefaultParamObject): parameter object to pass to generating function.
        verbose (bool): whether to print debug statements.
    Returns:
        (float): number of seconds required to generate a single sample, + two standard deviations.
    """
    try:
        per_save = Cache(os.getcwd(), verbose)[dataset][params._to_string()].time_per_save
    except FileNotFoundError:
        try:
            per_save = Cache(os.getcwd(), verbose)[dataset][0].time_per_save
            print("No time data is stored for this parameter set.")
            yes = input("Would you like to use time data from another parameter set? (Y/n): ")
            if yes.lower() in {"n", "no"}:
                raise ValueError("no time data for this param set; --time must be provided")
        except FileNotFoundError:
            raise ValueError("no time data for this dataset; --time must be provided")
    utils.v_print(verbose, "two standard deviations above the mean is {}s".format(per_save))

    per_sample = per_save / dataset.cache_every
    utils.v_print(verbose, "\tper sample: {}s".format(per_sample))

    return per_sample


def samples_to_jobs(size, njobs):
    """Return a list of sample sizes for each job, by assigning as evenly as possible.

    Args:
        size (int): number of samples requested.
        njobs (int): number of jobs to create.
    Returns:
        (list): list of length `njobs`, containing the number of samples for each job to
                generate.
    """
    even = size // njobs
    out = [even] * njobs
    for i in range(size - even * njobs):  # add remainder evenly at the front
        out[i] += 1
    return out


def test_samples_to_jobs():
    """Run simple test cases against samples_to_jobs."""
    assert samples_to_jobs(100, 1) == [100]
    assert samples_to_jobs(100, 2) == [50, 50]
    assert samples_to_jobs(100, 3) == [34, 33, 33]
    assert samples_to_jobs(101, 3) == [34, 34, 33]


def _generate_with_SLURM(dataset, size, params, njobs=1, job_time=None, verbose=False):
    """Using SLURM, generate `size` samples of the dataset.

    Args:
        dataset (class): A dataset from datasets.py.
        size (int): number of samples to generate.
        params (utils.DefaultParamObject): parameter object to pass to generating function.
        njobs (int): number of SLURM jobs to use to generate the data.
        job_time (str): time for each batch job, or None if adapted from metrics.
        verbose (bool): whether to print debug statements.
    """
    _remove_metadata(dataset, params, verbose)

    SLURM_out = os.path.join(os.getcwd(), ".slurm_cache", ".slurm_output")
    os.makedirs(SLURM_out, exist_ok=True)

    per_sample = None
    if job_time is None:
        per_sample = _get_sample_time(dataset, params, verbose)

    for nsamples in samples_to_jobs(size, njobs):
        # nsamples is the number of samples each job should generate

        if per_sample is not None:
            # set the amount of time for this job depending on the number of samples
            # round up to the next minute
            job_time = int(per_sample * nsamples / 60) + 1

        command = create_SLURM_command(dataset, nsamples, params, SLURM_out, job_time)
        utils.v_print(verbose, "Submitting the following job:")
        utils.v_print(verbose, command)

        try:
            process = subprocess.run(command, shell=True, universal_newlines=True, check=True)

            if process.stdout is not None or process.stderr is not None:
                print("Stdout:", process.stdout, sep="\n")
                print("Stderr:", process.stderr, sep="\n")
        except subprocess.CalledProcessError:
            raise RuntimeError("error encountered while submitting SLURM job")


# These are exceptions that should cause an error report to the user instead of a full exception
# message.
known_exceptions = [
    "no time data is stored for this dataset; --time must be provided",
    "no dataset file found",
]


def generate(dataset, n, params, njobs=1, job_time=None, this_node=False, verbose=False):
    """Submit the SLURM jobs to create the samples requested from the command line.

    Args:
        dataset (str): name of the data generating function. The function must be defined in a file
                       named `datasets.py` in the current directory, and must have the
                       `slurm_gen.dataset` decorator.
        n (int): number of data points to generate.
        params (dict): parameters to be passed to the data generator.
        njobs (int): number of SLURM jobs to submit. Each job will produce around n / njobs samples.
        job_time (str): time for each job to run, or None for time adapted from metrics.
        this_node (bool): if True, the data is generated in the current process instead of on SLURM.
                          This ignores all SLURM options.
        verbose (bool): print debug-level information from all functions.
    """
    try:
        dataset = utils.get_dataset(dataset)
        params = dataset.param_class(**params)
        if this_node:
            _generate_here(dataset, n, params, verbose)
        else:
            _generate_with_SLURM(dataset, n, params, njobs, job_time, verbose)
    except ValueError as e:
        if str(e) in known_exceptions:
            print("Error: " + str(e))
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate some samples for a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required arguments
    parser.add_argument("dataset", help="dataset for which to generate samples")
    parser.add_argument("-n", type=int, required=True, help="number of data points to generate")
    parser.add_argument(
        "--njobs",
        type=int,
        required="--this-node" not in sys.argv,  # only require if submitting to SLURM
        help="number of SLURM jobs to submit. Each job will produce about n/njobs samples",
    )

    # optional arguments
    parser.add_argument(
        "--this-node",
        action="store_true",
        help="run the data-generating code in the current process, instead of submitting it as a "
        "SLURM job. This ignores all SLURM options.",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=ast.literal_eval,
        default={},
        help="dict to be passed as params to the dataset.",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        help='time for each job to run. Acceptable time formats include "MM", "MM:SS", "HH:MM:SS", '
        '"D-HH", "D-HH:MM" and "D-HH:MM:SS". If not provided, the second standard deviation above '
        "the mean of previous runs is used (adapted to the number of samples per job).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print debug-level information from all functions",
    )

    args = parser.parse_args()

    utils.gitHubIssueHandler(
        generate,
        args.dataset,
        args.n,
        args.params,
        args.njobs,
        args.time,
        args.this_node,
        args.verbose,
    )
