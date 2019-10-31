"""Assorted functions used for simulation and data preparation.

Kyle Roth. 2019-05-17."""


from glob import iglob as glob
import inspect
import os
import pickle
import time


def v_print(verbose, s):
    """If verbose is True, print the string, prepending with the current timestamp.

    Args:
        verbose (bool): whether to print.
        s (str): string to print. (This will be passed to str.format, so it could be
                 anything with a __repr__.)
    """
    if verbose:
        print(
            "{:.6f}: ({}) {}".format(
                time.time(), inspect.currentframe().f_back.f_code.co_name, s
            )
        )


def to_pickle(thing, filepath):
    """Write the thing to a pickle file at the path specified.

    Args:
        thing: pickleable object.
        filepath (str): path where file will be written.
    """
    with open(filepath, "wb") as f:
        pickle.dump(thing, f)


def from_pickle(filepath, verbose=False):
    """Load the thing from the pickle file at the path specified.

    Errors while reading the pickle result in deleting the file. It's better to remove
    the file and try to recover than to raise an exception simply because a data run
    resulted in a corrupted file.

    Args:
        filepath (str): path where the pickle is located.
        verbose (bool): whether to print debug statements.
    Returns:
        : the object stored in the pickle, or ([], []) if the read was unsuccessful.
    """
    try:
        with open(filepath, "rb") as f:
            out = pickle.load(f)
        v_print(verbose, "Cache file successfully loaded.")
        return out
    except (EOFError, pickle.UnpicklingError):
        v_print(verbose, "Cache file corrupt; deleting.")
        os.remove(filepath)
        return [], []


class InsufficientSamplesError(Exception):
    """Helpful error messages for determining how much more time is needed to create the
    requested size of dataset."""

    def __init__(self, size, dataset_path, verbose=False):
        """Use the size needed to determine how much compute time it will take to
        generate that many samples.

        Args:
            size (int): number of samples needed.
            dataset_path (str): path to the dataset. Should include params but not set
                                name ('train', 'test').
            verbose (bool): print debugging statements to stdout.
        """
        self.needed = size
        times = self._get_times(dataset_path, verbose)
        if times:
            est_time = self.s_to_clock(size * sum(times) / len(times))
            super(InsufficientSamplesError, self).__init__(
                "{} samples need to be generated (estimated time {})".format(
                    size, est_time
                )
            )
        else:
            # no times have been recorded yet
            super(InsufficientSamplesError, self).__init__(
                "{} samples need to be generated;"
                " generate a small number first to estimate time".format(size)
            )

    @staticmethod
    def _get_times(dp, verbose=False):
        """Get the recorded times for previous data generation.

        Also compile the times into a single file so it's faster next time.

        Args:
            dp (str): path to the dataset, including params but not the set name.
            verbose (bool): print debugging statements to stdout.
        Returns:
            list(float): the collected times, in seconds.
        """
        times = []
        os.makedirs(os.path.join(dp, ".times"), exist_ok=True)
        for fp in glob(os.path.join(dp, ".times", "*.time")):
            with open(fp, "r") as infile:
                times.extend(infile.read().strip().split())
            os.remove(fp)
        v_print(
            verbose,
            'Collected {} timings from "{}".'.format(
                len(times), os.path.join(dp, ".times")
            ),
        )

        # store in a single file
        with open(os.path.join(dp, ".times", "compiled.time"), "w") as outfile:
            outfile.writelines([time + "\n" for time in times])
        v_print(
            verbose,
            'Wrote timings to "{}"'.format(os.path.join(dp, ".times", "compiled.time")),
        )

        return [float(time) for time in times]

    @staticmethod
    def s_to_clock(s):
        """Convert seconds to clock format, rounding up.

        Args:
            s (float): number of seconds.
        Returns:
            (str): clock time in HH:MM:SS format.
        """
        s = int(s) + 1
        h = s // 3600
        s = s % 3600
        m = s // 60
        s = s % 60
        return "{}:{}:{}".format(str(h).zfill(2), str(m).zfill(2), str(s).zfill(2))


class DefaultParamObject:
    """Class with attributes specifying default parameters for experiments.

    Not useful to instantiate on its own.
    """

    def __init__(self, **kwargs):
        """Replace any default values with those specified in the constructor call.

        Args:
            kwargs: parameter values to replace defaults.
        """
        # ensure all kwargs are class attributes
        if not set(kwargs.keys()).issubset(self.__dict__):
            raise AttributeError(
                "the following parameters are not attributes of {}: {}".format(
                    type(self).__name__, set(kwargs.keys()) - set(self.__dict__)
                )
            )
        self.__dict__.update(kwargs)

    def _to_string(self):
        """Get a string representing the values of all the parameters.

        This is used to create the directory for samples created with these parameters.
        Attributes beginning with an underscore are not included.

        Returns:
            (str): each parameter's name and value, separated by pipe characters ("|").
        """
        out = ""
        for attr in dir(self):
            if not attr.startswith("_"):
                out += "|" + attr + repr(getattr(self, attr))

        return out[1:]  # cut off first pipe character


def get_func_name(func):
    """Get the name of the function as a string.

    For use when caching and loading preprocessed datasets.

    Args:
        func (callable): function.
    Returns:
        (str): __name__ attribute of func, unless func is None then 'none'.
    """
    if func is None:
        return "none"  # unprocessed version
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
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
    )


def get_dataset_dir(name, params):
    """Get the absolute path to the cache location for this specific dataset.

    Args:
        name (str): name of dataset.
        params: an object with a _to_string() method, containing parameters used by the
                generating function.
    """
    return os.path.join(get_cache_dir(), name, params._to_string())


def get_SLURM_output_dir():
    """Get the absolute path of the directory where data generation jobs should place
    their output.

    From the directory of this module, this is the absolute path of ./slurm_output.

    Returns:
        (str): the absolute path.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "slurm_output")


def get_unique_filename():
    """Get a string guaranteed not to be repeated on the same computer (unless the clock
    changes).

    If inside a SLURM job, return the SLURM job ID. Otherwise, return digits from
    time.time().

    Returns:
        (str): unique string.
    """
    if "SLURM_JOBID" in os.environ:
        return os.environ["SLURM_JOBID"]
    return str(time.time()).replace(".", "")


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


def _clock_to_seconds(s, val):
    """Helper function for clock_to_seconds.

    Args:
        s (str): substring containing values separated by colons.
        val (int): initial time multiplier, which is multiplied by 60 after each
                   iteration.
    Returns:
        (int): computed time value.
    """
    out = 0

    for count in reversed(s.split(":")):
        out += int(count) * val
        val *= 60

    return out


def clock_to_seconds(s):
    """Convert a time string (in one of the formats accepted by SLURM) to a number of
    seconds.

    Acceptable time formats include "MM", "MM:SS", "HH:MM:SS", "D-HH", "D-HH:MM" and
    "D-HH:MM:SS".

    Args:
        s (str): time string.
    Returns:
        (int): the number of seconds.
    """
    try:
        out = 0
        days = None

        # days
        if "-" in s:
            days, s = s.split("-")
            out += int(days) * 86400

            if len(s.split(":")) == 2:
                # HH:MM
                return out + _clock_to_seconds(s, 60)
            if len(s.split(":")) == 1:
                # HH
                return out + int(s) * 3600

        split = s.split(":")

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


def count_samples(path, verbose=False):
    """Count the number of samples in all the pickle files in the path.

    Pickle files end with ".pkl".

    Args:
        path (str): directory containing pickle files.
        verbose (bool): whether to print debug statements.
    """
    count = 0
    for name in os.listdir(path):
        full_name = os.path.join(path, name)
        if name.endswith(".pkl") and os.path.isfile(full_name):
            count += len(from_pickle(full_name, verbose)[1])
    return count


def get_count(path, verbose=False):
    """Get the quantity of data samples available at the set path.

    Looks for a metadata file named '.metadata', which should contain the size. If not,
    it collects the pickle files and determines the size.

    If the path is to a 'raw' set of samples, just the number of samples is returned.
    Otherwise, a dict mapping preprocessor names to numbers of samples is returned.

    Args:
        path (str): path to directory.
        verbose (bool): whether to print debug statements.
    Returns:
        (int or dict): number of samples.
    """
    # check to see if the .metadata file contains the size
    v_print(verbose, 'Getting count for directory "{}"'.format(path))
    if os.path.isfile(os.path.join(path, ".metadata")):
        v_print(verbose, "Using metadata file")
        with open(os.path.join(path, ".metadata"), "r") as metadata:
            if path.endswith("raw"):
                for line in metadata:
                    if line.startswith("size: "):
                        # for raw, there is no preprocessing: only a number is desired
                        out = int(line.split(": ")[1])
                        v_print(verbose, "Found raw size {}".format(out))
                        return out
            else:
                out = {}
                for line in metadata:
                    if line.startswith('size "'):
                        out[line.split('"')[1]] = int(line.split(": ")[1])
                v_print(verbose, "Found sizes: {}".format(out))
                return out

    # count the samples in the pkl files by hand
    print("No metadata is present for the dataset at {}".format(path))
    print(
        "We'll have to count by hand."
        " This may take a long time, depending on the number of samples."
    )
    v_print(verbose, "Counting by hand")
    if path.endswith("raw"):
        count = 0
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                v_print(verbose, 'Adding count from file "{}"'.format(file))
                count += len(from_pickle(os.path.join(path, file))[1])
                v_print(verbose, "Count is now {}".format(count))

        # save the count for next time
        v_print(
            verbose, 'Writing count to "{}"'.format(os.path.join(path, ".metadata"))
        )
        with open(os.path.join(path, ".metadata"), "w+") as metadata:
            metadata.write("size: {}\n".format(count))
    else:
        count = {}
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                v_print(verbose, 'Adding count from file "{}"'.format(file))
                count[file[:-4]] = len(from_pickle(os.path.join(path, file))[1])
                v_print(verbose, "Count is now {}".format(count))

        # save the count for next time
        v_print(
            verbose, 'Writing count to "{}"'.format(os.path.join(path, ".metadata"))
        )
        with open(os.path.join(path, ".metadata"), "w+") as metadata:
            for set_name in count:
                metadata.write('size "{}": {}\n'.format(set_name, count[set_name]))

    return count


def get_counts(dataset, verbose=False):
    """Get detailed quantity information for a dataset.

    A possible return dict could look like the following:

    {
        'some|param|options': {
            'raw': 500,
            'train': {'some_preprocessor': 1000, None: 1000},
            'val': {None: 500},
            'test': {None: 500}
        },
        'other|param|options': {
            'raw': 2000,
            'train': {},
            'val': {},
            'test': {}
        }
    }

    Args:
        dataset (str): name of dataset.
        verbose (bool): whether to print debug statements.
    Returns:
        (dict): a map from subset strings to counts, at the necessary depths.
    """
    dataset_dir = os.path.join(get_cache_dir(), dataset)
    v_print(verbose, 'Dataset directory: "{}"'.format(dataset_dir))
    out = {}

    for params in os.listdir(dataset_dir):
        out[params] = {}
        v_print(verbose, "Params: {}".format(params))
        # sort to maintain order and allow the user to select param sets by order
        for set_name in sorted(os.listdir(os.path.join(dataset_dir, params))):
            if not set_name.startswith("."):  # don't grab things like '.times'
                v_print(verbose, "Set name: {}".format(set_name))
                out[params][set_name] = get_count(
                    os.path.join(dataset_dir, params, set_name), verbose
                )

    return out
