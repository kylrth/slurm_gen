"""Assorted functions used for simulation and data preparation.

Kyle Roth. 2019-05-17."""


import inspect
import os
import time

from scipy.interpolate import interp1d
import numpy as np


def v_print(verbose, s):
    """If verbose is True, print the string, prepending with the current timestamp.

    Args:
        verbose (bool): whether to print.
        s (str): string to print. (This will be passed to str.format, so it could be anything with a __repr__.)
    """
    if verbose:
        print('{}: ({}) {}'.format(time.time(), inspect.currentframe().f_back.f_code.co_name, s))


def make_grid(xs, hs):
    """Creates a 2D grid with the density specified by xs and hs.

    Used by pyMode for simulation points.

    Args:
        xs (array-like): positions at which to switch densities.
        hs (array-like): densities to use between positions.
    Returns:
        (np.ndarray): linspace of points at the appropriate density for each section.
    """
    grid = [min(xs)]
    interp = interp1d(xs, hs, kind='linear')
    while grid[-1] < max(xs):
        h = interp(grid[-1])
        grid.append(grid[-1] + h)
    return np.array(grid)


class PyModeParamObject:
    """Class with attributes specifying parameters for PyMode experiments. Not useful to instantiate on its own."""
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


def get_dataset_dir(name, params):
    """Get the absolute path to the cache location for this specific dataset.

    The dataset directory is created if it does not exist.

    Args:
        name (str): name of dataset.
        params (dict): parameters used by the generating function.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')

    if params is None:
        dataset_dir = os.path.join(cache_dir, name)
    else:
        dataset_dir = os.path.join(cache_dir, name, dict_to_path(params))

    os.makedirs(dataset_dir, exist_ok=True)

    return dataset_dir


def get_unique_filename():
    """Get a string guaranteed not to be repeated on the same computer (unless the clock changes).

    If inside a SLURM job, return the SLURM job ID. Otherwise, return digits from time.time().

    Returns:
        (str): unique string.
    """
    if 'SLURM_JOBID' in os.environ:
        return os.environ['SLURM_JOBID']
    return str(time.time()).replace('.', '')
