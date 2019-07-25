"""Module to call to generate data.

python -m slurm_gen.generate dimension_sweep
python -m slurm_gen list
python -m slurm_gen dataset dimension_sweep list  # list params

Kyle Roth. 2019-07-24.
"""


import argparse


def main():
    """Submit the SLURM jobs to create the data.

    Args:
        p (argparse.ArgumentParser): argument parser object containing the attributes specified below in the
                                     "if __name__ == '__main__'" part.
    """
    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('dataset', type=str, required=True, help='dataset for which to generate samples')
    parser.add_argument('-n', type=int, required=True, help='number of data points to generate')
    parser.add_argument(
        '--mem_per_cpu', type=int, required=True, help='memory to assign to each CPU in a SLURM job, in GB')

    # optional arguments
    parser.add_argument(
        '-l', '--at_least', action='store_true',
        help='simply ensure that n data points are present, generating more if necessary')
    parser.add_argument(
        '-p', '--params', type=str,
        help='dict to be passed as params to the data generator. Use --describe to view the docstring for the '
        'generating function, which may describe possible keys and values for params.')
    parser.add_argument('--test', action='store_true', help='mark SLURM submissions with "--qos=test"')
    parser.add_argument(
        '-t', '--time', type=str,
        help='time for each job to run. Must match the format HH:MM:SS. If not specified, the third standard '
        'deviation above the mean of previous runs is used (adapted to the number of samples per job)')
    parser.add_argument('--cpus_per_task', type=int, default=1, help='number of CPUs per SLURM job')
    parser.add_argument('--name', type=str, default='data gen', help='name for SLURM task')
    parser.add_argument('--ntasks', type=int, default=1, help='number of tasks to assign to this job')
    parser.add_argument('--not_rhel7', action='store_true', help='don\'t specify the use of a RHEL7 image')
    parser.add_argument(
        '--preemptable', action='store_true',
        help='mark SLURM submissions with "--qos=standby", allowing preemption of the jobs')
    parser.add_argument('--GPUs', type=int, default=0, help='specify the number of GPUs to allocate to each job')

    parser.parse_args()

    main(parser)
