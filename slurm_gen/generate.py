"""Functionality for generating datasets defined in datasets.py.

Use these modules if you encounter a data_loading.InsufficientSamplesError.

This module provides a CLI, which is probably the easiest way to use it.

# TODO: move the following to the readme
Example commands to run from the command line:

python -m slurm_gen.generate dimension_sweep
python -m slurm_gen list
python -m slurm_gen dataset dimension_sweep list  # list params

Kyle Roth. 2019-05-27.
"""


import argparse
import os
import subprocess

from slurm_gen import utils

# the base string for submitting a data generation job
raw_SLURM = (
    """echo '#!/bin/bash

module load openblas
module load boost
module load python/3.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${{HOME}}/usr/lib/:${{HOME}}/usr/lib64/
export PATH="${{HOME}}/usr/bin:$PATH"

python3 -u -c"""
    """ "from {this_module}.datasets import {dataset} as gen; gen({size}, {params})"
' | sbatch --error="{out}/%j.out" --output="{out}/%j.out" --mem-per-cpu="{mem}" """
    """--cpus-per-task="{cpus}" -J "{name}" --ntasks="{ntasks}" --time="{time}" """
)


def create_SLURM_command(dataset, size, params, SLURM_out, options):
    """Create the SLURM command that will generate the requested number of samples.

    Args:
        dataset (str): name of dataset to generate. Must match a function from the
                       data_gen module.
        size (int): number of samples to generate.
        params (dict): parameter dict to pass to generating function. Must be
                       reconstructable from repr(params).
        SLURM_out (str): path to SLURM output directory.
        options: Object providing SLURM options in the following attributes:
                 - mem_per_cpu: memory to assign to each CPU in a SLURM job, e.g. "2GB".
                 - time: time for each job to run. Acceptable time formats include "MM",
                         "MM:SS", "HH:MM:SS", "D-HH", "D-HH:MM" and "D-HH:MM:SS". If
                         None, the third standard deviation above the mean of previous
                         runs is used (adapted to the number of samples per job).
                 - cpus_per_task: number of CPUs per SLURM task.
                 - name: name for each SLURM task.
                 - ntasks: number of tasks to assign to this job.
                 - test: mark SLURM submissions with "--qos=test".
                 - preemptable: mark SLURM submissions with "--qos=standby", allowing
                                preemption of the jobs.
                 - GPUs: specify the number of GPUs to allocate to each job.
    Returns:
        (str): viable bash command to submit SLURM job.
    """
    if options.test and utils.clock_to_seconds(options.time) > 3600:
        raise ValueError(
            "tests may not have a run time greater than 60 minutes on SLURM"
        )

    # insert options into command string
    command = raw_SLURM.format(
        this_module=os.path.basename(
            os.path.dirname(os.path.abspath(__file__))
        ),  # should be 'slurm_gen'
        dataset=utils.sanitize_str(dataset),
        size=utils.sanitize_str(size),
        params=utils.sanitize_str(repr(params)),
        out=utils.sanitize_str(SLURM_out),
        mem=utils.sanitize_str(options.mem_per_cpu),
        cpus=utils.sanitize_str(options.cpus_per_task),
        name=utils.sanitize_str(options.name),
        ntasks=utils.sanitize_str(options.ntasks),
        time=utils.sanitize_str(options.time),
    )

    # create QOS string
    qos = ""
    sep = " --qos="
    if options.test:
        qos += sep + "test"
        sep = ","
    if options.preemptable:
        qos += sep + "standby"

    if qos:
        command += qos

    if options.GPUs:
        command += " --gres=gpu:{}".format(utils.sanitize_str(options.GPUs))

    return command


def generate_data(dataset, size, params, options, njobs=1, verbose=False):
    """Using SLURM, generate `size` samples of the dataset, using the generation
    function defined in datasets.py.

    Args:
        dataset (str): name of dataset to generate. Must match a function retrieved by
                       get_dataset.
        size (int): number of samples to generate.
        params (dict): parameter dict to pass to generating function.
        options: object containing SLURM options as attributes. Required attributes are
                 listed in the docstring for create_SLURM_command.
        njobs (int): number of SLURM jobs to use to generate the data.
        verbose (bool): whether to print the bash commands issued.
    """
    SLURM_out = utils.get_SLURM_output_dir()

    for nsamples in utils.samples_to_jobs(size, njobs):
        # nsamples is the number of samples each job should generate
        command = create_SLURM_command(dataset, nsamples, params, SLURM_out, options)
        if verbose:
            print("Submitting the following job:")
            print(command)

        process = subprocess.run(
            command,
            cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ),  # be able to import the package
            shell=True,
            universal_newlines=True,
            check=True,
        )

        if process.stdout is not None or process.stderr is not None:
            print("Stdout:", process.stdout, sep="\n")
            print("Stderr:", process.stderr, sep="\n")


def main(p):
    """Submit the SLURM jobs to create the samples requested from the command line.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified
                                below in the parser definition.
    """
    generate_data(p.dataset, p.n, p.params, p, p.njobs, p.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate some datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required arguments
    parser.add_argument("dataset", help="dataset for which to generate samples")
    parser.add_argument(
        "-n", type=int, required=True, help="number of data points to generate"
    )
    parser.add_argument(
        "--njobs",
        type=int,
        required=True,
        help="number of SLURM jobs to submit. "
        "Each job will produce about n/njobs samples",
    )
    parser.add_argument(
        "--mem_per_cpu",
        type=str,
        required=True,
        help='memory to assign to each CPU in a SLURM job, e.g. "2GB"',
    )

    # optional arguments
    parser.add_argument(
        "-l",
        "--at_least",
        action="store_true",
        help="simply ensure that n data points are present, "
        "generating more if necessary",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default={},
        help="dict to be passed as params to the data generator. Use --describe to "
        "view the docstring for the generating function, which may describe possible "
        "keys and values for params.",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        help='time for each job to run. Acceptable time formats include "MM", "MM:SS", '
        '"HH:MM:SS", "D-HH", "D-HH:MM" and "D-HH:MM:SS". If not provided, the third '
        "standard deviation above the mean of previous runs is used (adapted to the "
        "number of samples per job).",
    )
    parser.add_argument(
        "--cpus_per_task", type=int, default=1, help="number of CPUs per SLURM task"
    )
    parser.add_argument(
        "--name", type=str, default="data gen", help="name for each SLURM task"
    )
    parser.add_argument(
        "--ntasks", type=int, default=1, help="number of tasks to assign to this job"
    )
    parser.add_argument(
        "--preemptable",
        action="store_true",
        help='mark SLURM submissions with "--qos=standby", allowing job preemption',
    )
    parser.add_argument(
        "--test", action="store_true", help='mark SLURM submissions with "--qos=test"'
    )
    parser.add_argument(
        "--GPUs",
        type=int,
        default=0,
        help="specify the number of GPUs to allocate to each job",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print debug-level information from all functions",
    )

    main(parser.parse_args())
