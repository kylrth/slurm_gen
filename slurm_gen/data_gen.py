"""Functionality for generating datasets defined in datasets.py.

Use these modules if you encounter a data_loading.InsufficientSamplesError.

Kyle Roth. 2019-05-27.
"""


import os
import subprocess
from .utils import sanitize_str


def samples_to_jobs(size, njobs):
    """Return a list of sample sizes for each job, by assigning as evenly as possible.

    Args:
        size (int): number of samples requested.
        njobs (int): number of jobs to create.
    Returns:
        (list): list of length `njobs`, containing the number of samples for each job to generate.
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


if __name__ == '__main__':
    test_samples_to_jobs()


SLURM_defaults = {
    'mem-per-cpu': 8,
    'cpus-per-task': 1,
    'rhel7': True,
    'ntasks': 1,
    'name': 'data gen',
    'preemptable': False,
    'test': False,
    'GPUs': 0
}


raw_SLURM = """echo '#!/bin/bash

module load openblas
module load boost
module load python/3.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${{HOME}}/usr/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${{HOME}}/usr/lib64/
export PATH="${{HOME}}/usr/bin:$PATH"

python3 -c "from time import time; from {}.datasets import {} as gen; gen({}, {})"
' | sbatch --error="{}/%j.out" --output="{}/%j.out" --mem-per-cpu="{}GB" --cpus-per-task="{}" -J "{}" """ \
"""--ntasks="{}" --time="{}" """


def create_SLURM_command(dataset, size, params, SLURM_out, options):
    """Create the SLURM command that will generate the requested number of samples.

    Available options include:
        'mem-per-cpu': memory per CPU (GB), default 16
        'cpus-per-task': number of CPUs used, default 1
        'rhel7': whether a RHEL7 node is requested, default True
        'ntasks': number of tasks to assign this job, default 1
        'name': name for SLURM submission, default 'data gen'
        'preemptable': whether to allow this job to be paused, default False
        'GPUs': number of GPUs to use, default 0

    Args:
        dataset (str): name of dataset to generate. Must match a function from the data_gen module.
        size (int): number of samples to generate.
        params (dict): parameter dict to pass to generating function. Must be reconstructable from repr(params).
        SLURM_out (str): path to SLURM output directory.
        options (dict): SLURM options; these override the defaults listed above.
    Returns:
        (str): viable bash command to submit SLURM job.
    """
    # set defaults
    defaults = SLURM_defaults
    # TODO: insert default time based on suggestion from previous runs
    defaults['time'] = '01:00:00'
    # update defaults with options
    defaults.update(options)

    if defaults['test'] and defaults['time'] > '01:00:00':
        raise ValueError('tests cannot be run for longer than 60 minutes on SLURM')

    # insert options into command string
    command = raw_SLURM.format(
        os.path.basename(os.path.dirname(os.path.abspath(__file__))),  # get name of module
        sanitize_str(dataset),
        sanitize_str(size),
        sanitize_str(repr(params)),
        sanitize_str(SLURM_out),
        sanitize_str(SLURM_out),
        sanitize_str(defaults['mem-per-cpu']),
        sanitize_str(defaults['cpus-per-task']),
        sanitize_str(defaults['name']),
        sanitize_str(defaults['ntasks']),
        sanitize_str(defaults['time'])
    )

    # add additional options
    if defaults['rhel7']:
        command += " -C 'rhel7'"
    if defaults['test']:
        command += ' --qos=test'
    elif defaults['preemptable']:  # the job script generator seems to reflect this functionality
        command += ' --qos=standby'
    if defaults['GPUs']:
        command += ' --gres=gpu:{}'.format(sanitize_str(defaults['GPUs']))

    return command


def generate_data(dataset, size, params, options=None, njobs=1, verbose=False):
    """Using SLURM, generate `size` samples of the dataset, using the generation function defined in datasets.py.

    Args:
        dataset (str): name of dataset to generate. Must match a function retrieved by get_dataset.
        size (int): number of samples to generate.
        params (dict): parameter dict to pass to generating function.
        options (dict): SLURM options. Defaults are listed in the docstring for create_SLURM_command.
        njobs (int): number of SLURM jobs to use to generate the data.
        verbose (bool): whether to print the bash commands issued.
    """
    if options is None:
        options = dict()

    # ensure SLURM output file directory exists
    SLURM_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slurm_output')
    os.makedirs(SLURM_out, exist_ok=True)

    for nsamples in samples_to_jobs(size, njobs):  # get the number of samples each job should generate
        command = create_SLURM_command(dataset, nsamples, params, SLURM_out, options)
        if verbose:
            print('Submitting the following job:')
            print(command)

        process = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # be able to import the package
            shell=True, universal_newlines=True, check=True)

        if process.stdout is not None or process.stderr is not None:
            print('Stdout:', process.stdout, sep='\n')
            print('Stderr:', process.stderr, sep='\n')
