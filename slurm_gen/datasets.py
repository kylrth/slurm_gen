"""Helper decorator that must be applied to data generating functions.

Each generating function should have call signature (size, params), where size is the number of
examples to produce and params is a dict of parameters to accept.

Kyle Roth. 2019-05-17.
"""


from functools import wraps
import os
import pickle
from time import time

from slurm_gen import utils


class objectless:
    """Base class for classes not to be instantiated."""
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("{} should not be instantiated".format(cls))


def dataset(ParamClass, mem, cache_freq, slurm_opts="", bash_cmds=None):
    """Create a decorator that turns a data generating function into one that stores data in the raw
    cache, caching every cache_every data points.

    Args:
        ParamClass (class): class name of the parameter object that the function accepts. Must have
                            a _to_string() method that creates a string of the parameter values, and
                            a constructor that creates a new object from a dict of params that
                            override defaults. It suffices to be a subclass of
                            utils.DefaultParamObject.
        mem (str): string describing the memory to assign to each CPU in a SLURM job, e.g. "2GB".
        cache_freq (int): number of data points to load between caching.
        slurm_opts (str): extra parameters to pass to the `sbatch` command when generating.
        bash_cmds (iterable(str)): lines of bash commands to insert in each job before calling the
                                   generator. A useful example is `module load python/3.6`.
    Returns:
        (decorator): decorator for a generating function that creates the dataset as described.
    """

    def decorator(f):
        """Turn a data generating function into one that stores data in the raw cache, caching every
        cache_every data points.

        Args:
            f (function): data generating function with call signature (size, params).
        Returns:
            (function): data generating function that stores the output of the original function in
                        the cache.
        """

        class Wrapper(objectless):
            """This class should remain static and replaces the function defined by the user in
            their dataset module."""

            # store the parameter class used by this function
            param_class = ParamClass

            # store how often the caching happens
            cache_every = cache_freq

            # store the SLURM batch options
            slurm_options = '--mem-per-cpu="{mem}" '.format(mem=mem) + slurm_opts

            # store the bash commands
            bash_commands = bash_cmds

            @classmethod
            def call(cls, size, params):
                """Write the results of the function to
                ./.slurm_cache/{dataset}/{params}/raw/{somefile}.pkl.

                Args:
                    size (int): number of samples to create.
                    params (dict): parameters to pass to generating function.
                """
                # convert to the parameter class
                params = cls.param_class(**params)

                # cache it in the raw location
                dataset_dir = utils.get_dataset_dir(os.getcwd(), f.__name__, params)
                unique_name = utils.get_unique_filename()
                raw_path = os.path.join(dataset_dir, "raw", unique_name + "_{}.pkl")
                print("Output path:", raw_path)
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)

                iter_data = f(size, params)

                data_count = 0
                to_store_x = []
                to_store_y = []

                # create a file to store times
                time_file = os.path.join(dataset_dir, "raw", ".times", "{}.time").format(
                    unique_name
                )
                os.makedirs(os.path.dirname(time_file), exist_ok=True)

                # do line buffering instead of file buffering, so that times are written in case the
                # process does not finish
                with open(time_file, "a+", buffering=1) as time_f:
                    while True:
                        start = time()

                        # try to get the next pair
                        try:
                            x, y = next(iter_data)
                        except StopIteration:
                            # store everything left over
                            if to_store_x:
                                with open(
                                    raw_path.format(data_count // cache_every + 1), "wb"
                                ) as pkl:
                                    pickle.dump((to_store_x, to_store_y), pkl)
                                time_f.write(str((time() - start) / cache_every) + "\n")
                            break

                        # add it to the temporary list
                        to_store_x.append(x)
                        to_store_y.append(y)
                        data_count += 1

                        # store it every `cache_every` iterations
                        if not data_count % cls.cache_every:
                            with open(raw_path.format(data_count // cls.cache_every), "wb") as pkl:
                                pickle.dump((to_store_x, to_store_y), pkl)

                            to_store_x.clear()
                            to_store_y.clear()

                            # record the average time spent
                            time_f.write(str((time() - start) / cls.cache_every) + "\n")

            @classmethod
            def preprocessor(cls, func):
                """Wrap a preprocessing function so it's available for this particular dataset.

                Args:
                    func (callable): preprocessing function.
                Returns:
                    (callable): wrapped function.
                """

                class Inner(objectless):
                    """This class should remain static and replaces the preprocessing function
                    defined by the user in their dataset module."""

                    # mark this preprocessor as a preprocessor for this dataset
                    dataset = cls

                    @staticmethod
                    def call(X, y):
                        """Call the preprocessor on some data.

                        Args:
                            X: features.
                            y: targets.
                        Returns:
                            X: modified(?) features.
                            y: modified(?) targets.
                        """
                        return func(X, y)

                Inner.__name__ = func.__name__
                Inner.__doc__ = func.__doc__
                return Inner

        Wrapper.__name__ = f.__name__
        Wrapper.__doc__ = f.__doc__
        return Wrapper

    return decorator
