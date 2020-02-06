"""Assorted functions used for simulation and data preparation.

Kyle Roth. 2019-05-17."""


from glob import iglob as glob
import importlib.util
import inspect
import os
import pickle
import platform
import sys
import time
import traceback
import urllib

import pkg_resources


def v_print(verbose, s):
    """If verbose is True, print the string, prepending with the current timestamp.

    Args:
        verbose (bool): whether to print.
        s (str): string to print. (This will be passed to str.format, so it could be
                 anything with a __repr__.)
    """
    if verbose:
        print(
            "{:.6f}: ({}) {}".format(time.time(), inspect.currentframe().f_back.f_code.co_name, s)
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
        v_print(verbose, "successfully read pickled object from '{}'".format(filepath))
        return out
    except (EOFError, pickle.UnpicklingError):
        v_print(verbose, "Cache file corrupt; deleting '{}'".format(filepath))
        os.remove(filepath)
        return [], []


class DefaultParamObject:
    """Class with attributes specifying default parameters for experiments.

    Not useful to instantiate on its own.
    """

    def __init__(self, **kwargs):
        """Replace any default values with those specified in the constructor call.

        Args:
            kwargs: parameter values to replace defaults.
        """
        # get the set of class attributes
        attrs = inspect.getmembers(self.__class__, lambda a: not (inspect.isroutine(a)))
        attrs = {a[0] for a in attrs if not (a[0].startswith("__") and a[0].endswith("__"))}

        # ensure all kwargs are class attributes
        if not set(kwargs.keys()).issubset(attrs):
            raise AttributeError(
                "the following parameters are not attributes of {}: {}".format(
                    type(self).__name__, set(kwargs.keys()) - attrs
                )
            )
        self.__dict__.update(kwargs)

    def _to_string(self):
        """Get a string representing the values of all the parameters.

        This is used to create the directory for samples created with these parameters.
        Attributes beginning with an underscore are not included.

        Returns:
            (str): each parameter's name and value divided by "#", with each pair separated by "|".
        """
        out = ""
        for attr in dir(self):
            if not attr.startswith("_"):
                out += "|{}#{}".format(attr, repr(getattr(self, attr)))

        return out[1:]  # cut off first pipe character

    @classmethod
    def _from_string(cls, s):
        """Create a new param object from the string produced by `_to_string()`.

        Args:
            s (str): string with "|" separating attributes and "#" dividing names and values.
        """
        args = {}
        for pair in s.split("|"):
            attr, val = pair.split("#")
            # convert to the type of the default attribute
            args[attr] = type(getattr(cls, attr))(val)
        return cls(**args)


def _import_datasets(path):
    """Import the datasets.py module defined in the given directory.

    Args:
        path (str): directory containing datasets.py.
    Returns:
        the imported module
    """
    # confirm the file exists
    if not os.path.isfile(os.path.join(path, "datasets.py")):
        raise ValueError("no dataset file found")

    spec = importlib.util.spec_from_file_location("datasets", os.path.join(path, "datasets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def is_dataset(obj):
    """Determine whether the object is a dataset wrapped with @slurm_gen.dataset.

    Args:
        obj
    Returns:
        (bool): whether it is a data generating function.
    """
    if not hasattr(obj, "call"):
        return False
    if not hasattr(obj, "param_class"):
        return False
    if not hasattr(obj, "slurm_options"):
        return False
    if not hasattr(obj, "cache_every"):
        return False
    return hasattr(obj, "bash_commands")


def get_datasets(path="."):
    """Return the datasets defined in the `datasets.py` module in the given directory.

    Args:
        path (str): directory to search.
    Returns:
        (list(callable)): generating functions.
    """
    mod = _import_datasets(path)

    # collect the datasets
    out = []
    for var in dir(mod):
        if is_dataset(getattr(mod, var)):
            out.append(getattr(mod, var))

    return out


def get_dataset(name, path=os.getcwd()):
    """Get the dataset by name as defined in the `datasets.py` module in the given directory.

    Args:
        name (str): name of function.
        path (str): directory to search.
    Returns:
        (class): dataset.
    """
    mod = _import_datasets(path)

    # ensure it's a generator
    if not is_dataset(getattr(mod, name)):
        raise ValueError(
            "no dataset '{}' found; was it wrapped with @slurm_gen.dataset?".format(name)
        )

    return getattr(mod, name)


def is_preprocessor(obj):
    """Determine whether the object is a preprocessor wrapped with @d.preprocessor where d is some
    dataset.

    Args:
        obj
    Returns:
        (bool): whether it is a data generating function.
    """
    if not hasattr(obj, "call"):
        return False
    if not hasattr(obj, "datasets"):
        return False
    if not isinstance(obj.datasets, set):
        return False
    for dataset in obj.datasets:
        if not is_dataset(dataset):
            return False
    return True


def get_preprocessors(dataset, path=os.getcwd()):
    """Return the preprocessors available to this dataset.

    Args:
        dataset (class): dataset decorated with @slurm_gen.dataset.
        path (str): path to root dataset directory.
    Returns:
        (list(class)): preprocessors.
    """
    mod = _import_datasets(path)

    # collect the preprocessors, filtering for the given dataset
    out = []
    for var in dir(mod):
        if is_preprocessor(getattr(mod, var)):
            out.append(getattr(mod, var))

    return out


def get_preprocessor(name, dataset, path=os.getcwd()):
    """Get the preprocessor by name as defined in the `datasets.py` module in the given directory.

    Args:
        name (str): name of the preprocessor.
        dataset (class): dataset decorated with @slurm_gen.dataset.
    Returns:
        (class): preprocessor.
    """
    mod = _import_datasets(path)

    if not is_preprocessor(getattr(mod, name)):
        raise ValueError(
            "no preprocessor '{p}' found for '{d}'; was it wrapped with @{d}.preprocessor?".format(
                p=name, d=dataset.__name__
            )
        )

    return getattr(mod, name)


def paramSetIdentifier(arg):
    """Convert the argument string to a unique identifier of a parameter set.

    Possible arguments are:
    - a whole number identifier (0, 1, 2, ...)
    - a string that ast.literal_eval can convert to a dictionary
      (e.g. "{'left': 0, 'std_dev': 0.5}")
    - a string created by a ParamObject (e.g. "left#0|right#1|std_dev#0.5")
    """
    # whole number identifier
    try:
        return int(arg)
    except ValueError:
        pass

    # dictionary
    try:
        return ast.literal_eval(arg)
    except ValueError:
        pass

    # ParamObject string
    dataset = utils.get_dataset(sys.argv[1])
    return dataset.paramClass._from_string(arg)


_github_issue_body = """PLEASE: Fill in a descriptive title and then delete this sentence.
PLEASE: Then add any helpful commentary before submitting.

Issue type: Exception

Command run:
```
{cmd}
```

Error:
```
Traceback (most recent call last):
{tb}
{err_msg}
```

SLURM-gen version: `{version}`
OS: `{os}`
"""


def createGitHubLink(args, tb, err_msg):
    """Return a link to create a new issue on SLURM-gen's issue tracker on GitHub, based on the
    error encountered.

    Args:
        args (list(string)): list of command-line arguments run to produce the exception.
        tb (string): traceback extracted from the error.
        err_msg (string): error message, including error type
                          (e.g. "ZeroDivisionError: division by zero").
    Returns:
        (string): link.
    """
    body = _github_issue_body.format(
        cmd=" ".join("'{}'".format(arg) for arg in args),
        version=pkg_resources.require("SLURM_gen")[0].version,
        os=platform.system(),
        tb=tb,
        err_msg=err_msg,
    )

    fields = {"body": body, "labels": ",".join(["bug"])}
    return "https://github.com/kylrth/slurm_gen/issues/new?" + urllib.parse.urlencode(fields)


def gitHubIssueHandler(func, *args, **kwargs):
    """Call the function with the provided args and kwargs, but wrap exceptions with a link to open
    a related GitHub issue for the project.

    Args:
        func (callable): function to call on args and kwargs.
        args
        kwargs
    Returns:
        anything the function call returns.
    """
    try:
        func(*args, **kwargs)
    except KeyboardInterrupt:
        print("\nexiting")
    except Exception as e:
        tb = "".join(traceback.format_tb(e.__traceback__)).rstrip()
        err_msg = repr(e)
        args = sys.argv
        args[0] = "slurm_gen" + os.path.sep + args[0].split("slurm_gen" + os.path.sep)[-1]

        print("Traceback (most recent call last):")
        print(tb)
        print(err_msg)
        print("\nTo submit an issue to GitHub, open the following link:\n")
        print(createGitHubLink(sys.argv, tb, err_msg))
