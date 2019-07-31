"""Assorted functions used for simulation and data preparation.

Kyle Roth. 2019-05-17."""


import inspect
import os
import pickle
import time


def v_print(verbose, s):
    """If verbose is True, print the string, prepending with the current timestamp.

    Args:
        verbose (bool): whether to print.
        s (str): string to print. (This will be passed to str.format, so it could be anything with a __repr__.)
    """
    if verbose:
        print('{:.6f}: ({}) {}'.format(time.time(), inspect.currentframe().f_back.f_code.co_name, s))


def to_pickle(thing, filepath):
    """Write the thing to a pickle file at the path specified.

    Args:
        thing: pickleable object.
        filepath (str): path where file will be written.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(thing, f)


def from_pickle(filepath):
    """Load the thing from the pickle file at the path specified.

    Args:
        filepath (str): path where the pickle is located.
    Returns:
        : the object stored in the pickle.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class CacheMutex:
    """Places and manages a lock file on data in a raw directory, so that concurrent threads don't assume access to the
    same pickle files."""
    def __init__(self, locked_path, verbose=False):
        """Ensure the path is available for locking, and lock it.

        Args:
            locked_path (str): path to directory to lock.
            verbose (bool): whether to print debug statements.
        """
        self.mutex_path = os.path.join(locked_path, 'cachemut.ex')
        self.verbose = verbose
        v_print(self.verbose, 'Mutex path: "{}"'.format(self.mutex_path))

    def __enter__(self):
        """Wait for possession the mutex."""
        os.makedirs(os.path.dirname(self.mutex_path), exist_ok=True)

        # wait for mutex to be available
        v_print(self.verbose, 'Awaiting mutex.')
        while os.path.isfile(self.mutex_path):
            time.sleep(1)

        # lock the mutex
        open(self.mutex_path, 'w').close()
        v_print(self.verbose, 'Mutex locked.')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the mutex."""
        os.remove(self.mutex_path)
        v_print(self.verbose, 'Mutex released.')


class DefaultParamObject:
    """Class with attributes specifying default parameters for experiments. Not useful to instantiate on its own."""
    def __init__(self, **kwargs):
        """Replace any default values with those specified in the constructor call.

        Args:
            kwargs: parameter values to replace defaults.
        """
        # ensure all kwargs are class attributes
        if not set(kwargs.keys()).issubset(self.__dict__):
            raise AttributeError(
                'the following parameters are not attributes of {}: {}'.format(
                    type(self).__name__,
                    set(kwargs.keys()) - set(self.__dict__)
                ))
        self.__dict__.update(kwargs)

    def to_string(self):
        """Get a string representing the values of all the parameters.

        This is used to create the directory for samples created with these parameters. Attributes beginning with an
        underscore are not included.

        Returns:
            (str): the printable representation (repr) of each parameter, separated by pipe characters ("|").
        """
        out = ""
        for attr in dir(self):
            if not attr.startswith('_'):
                out += '|' + repr(getattr(self, attr))

        return out[1:]  # cut off first pipe character


def dict_to_path(d):
    """Create a string of the values of the dictionary, separated by dashes.

    Args:
        d (dict).
    Returns:
        (str): The dictionary's values separated by dashes.
    """
    return '-'.join(str(val) for val in d.values())


def get_func_name(func):
    """Get the name of the function as a string.

    For use when caching and loading preprocessed datasets.

    Args:
        func (callable): function.
    Returns:
        (str): __name__ attribute of func, unless func is None then 'none'.
    """
    if func is None:
        return 'none'  # unprocessed version
    return func.__name__


def sanitize_str(s):
    """Sanitize strings to send to a shell.

    Args:
        s (str): string (or something to be converted to a string).
    Returns:
        (str): sanitized string.
    """
    return str(s).replace('"', '\\"')


def get_cache_dir():
    """Get the absolute path to the cache location.

    Returns:
        (str): the absolute path.
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')


def get_dataset_dir(name, params):
    """Get the absolute path to the cache location for this specific dataset.

    The dataset directory is created if it does not exist.

    Args:
        name (str): name of dataset.
        params (dict): parameters used by the generating function.
    """
    if params is None:
        dataset_dir = os.path.join(get_cache_dir(), name)
    else:
        dataset_dir = os.path.join(get_cache_dir(), name, dict_to_path(params))

    os.makedirs(dataset_dir, exist_ok=True)

    return dataset_dir


def get_SLURM_output_dir():
    """Get the absolute path of the directory where data generation jobs should place their output.

    From the directory of this module, this is the absolute path of ./slurm_output.

    Returns:
        (str): the absolute path.
    """
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slurm_output')
    os.makedirs(out, exist_ok=True)
    return out


def get_unique_filename():
    """Get a string guaranteed not to be repeated on the same computer (unless the clock changes).

    If inside a SLURM job, return the SLURM job ID. Otherwise, return digits from time.time().

    Returns:
        (str): unique string.
    """
    if 'SLURM_JOBID' in os.environ:
        return os.environ['SLURM_JOBID']
    return str(time.time()).replace('.', '')


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


def _clock_to_seconds(s, val):
    out = 0

    for count in reversed(s.split(':')):
        out += int(count) * val
        val *= 60

    return out


def clock_to_seconds(s):
    """Convert a time string (in one of the formats accepted by SLURM) to a number of seconds.

    Acceptable time formats include "MM", "MM:SS", "HH:MM:SS", "D-HH", "D-HH:MM" and "D-HH:MM:SS".

    Args:
        s (str): time string.
    Returns:
        (int): the number of seconds.
    """
    try:
        out = 0
        days = None

        # days
        if '-' in s:
            days, s = s.split('-')
            out += int(days) * 86400

            if len(s.split(':')) == 2:
                # HH:MM
                return out + _clock_to_seconds(s, 60)
            if len(s.split(':')) == 1:
                # HH
                return out + int(s) * 3600

        split = s.split(':')

        if len(split) == 3:
            # HH:MM:SS
            return out + _clock_to_seconds(s, 1)

        if len(split) == 2:
            # MM:SS
            return _clock_to_seconds(s, 1)

        if len(split) == 1:
            ## MM
            return int(s) * 60

        raise ValueError()
    except ValueError:
        raise ValueError('clock string has bad formatting: "{}"'.format(s))

