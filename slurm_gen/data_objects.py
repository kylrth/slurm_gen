"""Objects for representing and managing datasets in the cache.

The hierarchy is Cache -> Dataset -> ParamSet -> Group -> PreprocessedGroup

Kyle Roth. 2019-08-05.
"""


import os
import statistics

from slurm_gen import utils


class PreprocessedData:
    """Represents a single list of data samples, preprocessed by a specific
    preprocessor."""

    def __init__(self, cache_path, verbose=False):
        """Store the path to the data.

        Args:
            cache_path (str): path to the directory containing pickle files with the
                              processed data.
            verbose (bool): whether to print debug statements.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new PreprocessedData object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

    @property
    def size(self):
        """The size of the preprocessed set of data.

        Returns:
            (int)
        """
        # try to read it from the metadata file
        metadata_path = os.path.join(self.path, ".metadata")
        if os.path.isfile(metadata_path):
            utils.v_print(self.verbose, "reading size from '{}'".format(metadata_path))
            with open(metadata_path) as infile:
                for line in infile:
                    if line.startswith("size: "):
                        self._size = int(line.split(": ")[1])
                        return self._size

        # count the number by hand
        self._size = utils.count_samples(self.path, self.verbose)

        # store it in the metadata file
        utils.v_print(self.verbose, "writing size '{}' to '{}'".format(self._size, metadata_path))
        with open(metadata_path, "a+") as outfile:
            outfile.write("size: {}\n".format(self._size))

        return self._size

    def get(self, size):
        """Return `size` samples from this preprocessed dataset.

        Warning: this data comes in order. You'll probably want to shuffle it.

        Args:
            size (int): number of samples to return.
        Returns:
            tuple(list, list): samples.
        """
        return utils.get_samples(self.path, size, self.verbose)


class Group:
    """Represents a group of data samples, and contains references to all contained
    PreprocessedData."""

    def __init__(self, cache_path, verbose=False):
        """Store the path to the group, to allow access to raw data.

        Args:
            cache_path (str): path to group; should contain pickle files for the
                              unprocessed set and directories for each preprocessed set.
            verbose (bool): whether to print debug statements.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new Group object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._iterator = None

    @property
    def unprocessed_size(self):
        """The size of the unprocessed set of data.

        Returns:
            (int)
        """
        # try to read it from the metadata file
        metadata_path = os.path.join(self.path, ".metadata")
        if os.path.isfile(metadata_path):
            utils.v_print(self.verbose, "reading size from '{}'".format(metadata_path))
            with open(metadata_path) as infile:
                for line in infile:
                    if line.startswith("size:"):
                        self._unprocessed_size = int(line.split(": ")[1])
                        return self._unprocessed_size

        # count the number by hand
        self._unprocessed_size = utils.count_samples(self.path, self.verbose)

        # store it in the metadata file
        utils.v_print(
            self.verbose, "writing size '{}' to '{}'".format(self._unprocessed_size, metadata_path)
        )
        with open(metadata_path, "a+") as outfile:
            outfile.write("size: {}\n".format(self._unprocessed_size))

        return self._unprocessed_size

    def get(self, size):
        """Get `size` samples from the unprocessed set.

        Warning: this data comes in order. You'll probably want to shuffle it.

        Args:
            size (int): number of samples to retrieve.
        Returns:
            tuple(list, list): samples.
        """
        return utils.get_samples(self.path, size, self.verbose)

    def assert_preprocessed_size(self, func, size):
        """Ensure that at least `size` samples have been processed with `func`.

        The preprocessor must accept a list of data points and a list of corresponding labels. It
        should return preprocessed versions of both data and label.

        Args:
            func (callable): preprocessing function to be applied to data. `func` is called on X, y
                             where X is an iterable of all data points and y is an iterable of all
                             labels.
            size (int): number of samples to preprocess.
        Raises:
            (utils.InsufficientSamplesError): there are not enough unprocessed samples.
        """
        if self.unprocessed_size < size:
            raise utils.InsufficientSamplesError(
                size - self.unprocessed_size, self.path, self.verbose
            )

        # get unprocessed data
        unproc_files = sorted(file for file in os.listdir(self.path) if file.endswith(".pkl"))

        # make sure the PreprocessedData exists
        os.makedirs(os.path.join(self.path, func.__name__), exist_ok=True)
        proc_set = self[func.__name__]

        # get the current size of the PreprocessedData and the individual files which
        # have already been preprocessed
        count = proc_set.size
        current_proc_files = set(os.listdir(proc_set.path))

        # preprocess files, skipping those that are already preprocessed
        for file in unproc_files:
            if count >= size:
                break
            if file in current_proc_files:
                continue

            # preprocess it!
            X, y = utils.from_pickle(os.path.join(self.path, file), self.verbose)
            count += len(X)
            X, y = func(X, y)
            utils.to_pickle((X, y), os.path.join(proc_set.path, file))

        # invalidate the metadata file
        try:
            os.remove(os.path.join(proc_set.path, ".metadata"))
        except FileNotFoundError:
            pass

    def __iter__(self):
        self._iterator = iter(
            {item for item in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, item))}
        )
        return self

    def __next__(self):
        """Iterate over the PreprocessedData for this Group.

        Returns:
            (PreprocessedData): the next preprocessed data that has been created under
                                this group.
        """
        return PreprocessedData(os.path.join(self.path, next(self._iterator)), self.verbose)

    def __getitem__(self, idx):
        """Allow access by name of PreprocessedData."""
        if not isinstance(idx, str):
            raise IndexError(
                "Preprocessed set access is only permitted by str; got {}".format(type(idx))
            )
        elements = os.listdir(self.path)
        if not os.path.isdir(os.path.join(self.path, idx)) or idx not in elements:
            raise IndexError(
                "no such preprocessed set found: {}".format(os.path.join(self.path, idx))
            )
        return PreprocessedData(os.path.join(self.path, idx), self.verbose)


class ParamSet:
    """Represents a parameter set with all its datasets beneath it."""

    def __init__(self, cache_path, verbose=False):
        """Store the path to the param set to allow access to child Groups and raw data.

        Args:
            cache_path (str): path to parameter set; should be a subdirectory of a
                              dataset directory.
            verbose (bool): whether to print debug statements.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new ParamSet object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._iterator = None

    @property
    def raw_size(self):
        """The size of the raw set of data.

        Returns:
            (int)
        """
        # try to read it from the metadata file
        raw_metadata_path = os.path.join(self.path, "raw", ".metadata")
        if os.path.isfile(raw_metadata_path):
            utils.v_print(self.verbose, "reading size from '{}'".format(raw_metadata_path))
            with open(raw_metadata_path) as infile:
                for line in infile:
                    if line.startswith("size: "):
                        self._raw_size = int(line.split(": ")[1])
                        return self._raw_size

        # count the number by hand
        self._raw_size = utils.count_samples(os.path.join(self.path, "raw"), self.verbose)

        # store it in the metadata file
        utils.v_print(
            self.verbose, "writing size '{}' to '{}'".format(self._raw_size, raw_metadata_path)
        )
        with open(raw_metadata_path, "w+") as outfile:
            outfile.write("size: {}\n".format(self._raw_size))

        return self._raw_size

    @property
    def time_per_save(self):
        """Two standard deviations above the mean amount of time it takes to generate `cache_every`
        samples.
        """
        times = []
        for file in os.listdir(os.path.join(self.path, "raw", ".times")):
            with open(os.path.join(self.path, "raw", ".times", file), "r") as timefile:
                for line in timefile:
                    times.append(float(line[:-1]))  # trailing newline

        return statistics.mean(times) + 2 * statistics.stdev(times)

    def move(self, target, size):
        """Move `size` raw samples to the group named `target`.

        Args:
            target (str): name of target group, e.g. "train" or "test".
            size (int): number of samples to move.
        """
        if target == "raw":
            raise ValueError("samples are already in 'raw' directory")

        if size > self.raw_size:
            raise utils.InsufficientSamplesError(size - self.raw_size, self.path)

        # make sure the group exists
        os.makedirs(os.path.join(self.path, target), exist_ok=True)

        # source and destination paths
        raw_path = os.path.join(self.path, "raw")
        target_path = os.path.join(self.path, target)
        utils.v_print(self.verbose, "source path: '{}'".format(raw_path))
        utils.v_print(self.verbose, "target path: '{}'".format(target_path))

        files = iter(file for file in os.listdir(raw_path) if not file.startswith("."))

        count = 0
        while True:
            current_file = next(files)
            X, y = utils.from_pickle(os.path.join(raw_path, current_file), self.verbose)
            count += len(X)
            if count > size:
                # split the file
                utils.v_print(
                    self.verbose,
                    "({}/{}) splitting file with {} samples into {} and {}".format(
                        count, size, len(X), count - size, len(X) - count + size
                    ),
                )
                utils.to_pickle(
                    (X[: size - count], y[: size - count]),
                    os.path.join(target_path, current_file[:-4] + "_0.pkl"),
                )
                utils.to_pickle(
                    (X[size - count :], y[size - count :]),
                    os.path.join(raw_path, current_file[:-4] + "_1.pkl"),
                )
                break
            else:
                utils.v_print(
                    self.verbose,
                    "({}/{}) moving file with {} samples into '{}'".format(
                        count, size, len(X), target
                    ),
                )
                os.rename(
                    os.path.join(raw_path, current_file), os.path.join(target_path, current_file)
                )
                if count == size:
                    break

        # invalidate the metadata files
        try:
            os.remove(os.path.join(self.path, "raw", ".metadata"))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.join(self.path, target, ".metadata"))
        except FileNotFoundError:
            pass

    def __iter__(self):
        self._iterator = iter(
            {item for item in os.listdir(self.path) if not item.startswith(".") and item != "raw"}
        )
        return self

    def __next__(self):
        """Iterate over the Groups for this ParamSet.

        Returns:
            (Group): the next group that has been created under this parameter set.
        """
        return Group(os.path.join(self.path, next(self._iterator)), self.verbose)

    def __getitem__(self, idx):
        """Allow access by name of Group."""
        if not isinstance(idx, str):
            raise IndexError("Group access is only permitted by str; got {}".format(type(idx)))

        if idx.startswith(".") or idx == "raw":
            raise IndexError("no such group found: {}".format(os.path.join(self.path, idx)))
        if idx not in os.listdir(self.path):
            raise IndexError("no such group found: {}".format(os.path.join(self.path, idx)))

        return Group(os.path.join(self.path, idx), self.verbose)

    def __contains__(self, item):
        return not item.startswith(".") and item != "raw" and item in os.listdir(self.path)


class Dataset:
    """Represents a dataset with all its parameter sets beneath it."""

    def __init__(self, cache_path, verbose=False):
        """Store the path to the dataset to allow access to child ParamSets.

        Args:
            cache_path (str): path to dataset; should be a subdirectory of the cache
                              directory.
            verbose (bool): whether to print debug statements.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new Dataset object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._iterator = None

    def __iter__(self):
        self._iterator = iter(sorted(os.listdir(self.path)))
        return self

    def __next__(self):
        """Iterate over the ParamSets for this dataset.

        Returns:
            (ParamSet): the next parameter set that has been generated for this dataset.
        """
        return ParamSet(os.path.join(self.path, next(self._iterator)), self.verbose)

    def __getitem__(self, idx):
        """Access either by name of ParamSet or by index."""
        elements = list(sorted(os.listdir(self.path)))
        if isinstance(idx, int):
            return ParamSet(os.path.join(self.path, elements[idx]), self.verbose)
        if isinstance(idx, str):
            if idx not in elements:
                raise ValueError(
                    "no such parameter set found: {}".format(os.path.join(self.path, idx))
                )
            return ParamSet(os.path.join(self.path, idx), self.verbose)
        raise IndexError(
            "ParamSet access is only permitted with int or str; got {}".format(type(idx))
        )


class Cache:
    """Represents a cache location and all of its datasets."""

    def __init__(self, verbose=False):
        """For now, the cache location is constant.

        Args:
            verbose (bool): whether to print debug statements.
        """
        self.path = utils.get_cache_dir()
        self.verbose = verbose
        utils.v_print(verbose, "creating new Cache object with path '{}'".format(self.path))

        self._iterator = None

    def __iter__(self):
        self._iterator = iter(
            sorted(item for item in os.listdir(self.path) if not item.startswith("."))
        )
        return self

    def __next__(self):
        """Iterate over datasets in the cache.

        Returns:
            (Dataset): the next dataset in the cache.
        """
        return Dataset(os.path.join(self.path, next(self._iterator)), self.verbose)

    def __getitem__(self, idx):
        """Access either by name of Dataset or by index."""
        elements = list(sorted(item for item in os.listdir(self.path) if not item.startswith(".")))
        if isinstance(idx, int):
            return Dataset(os.path.join(self.path, elements[idx]), self.verbose)
        if isinstance(idx, str):
            if idx not in elements:
                raise ValueError("no such dataset found: {}".format(os.path.join(self.path, idx)))
            return Dataset(os.path.join(self.path, idx), self.verbose)
        raise IndexError(
            "Dataset access is only permitted with int or str; got {}".format(type(idx))
        )
