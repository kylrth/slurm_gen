"""Objects for representing and managing datasets in the cache.

The hierarchy is Cache -> Dataset -> ParamSet -> Group -> PreprocessedGroup

Kyle Roth. 2019-08-05.
"""


import os
import statistics

from slurm_gen import datasets, utils


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
        try:
            dataset = os.path.basename(os.path.dirname(os.path.normpath(dataset_path)))
            cache_every = getattr(datasets, dataset).cache_every
            time_per_sample = ParamSet(dataset_path, verbose).time_per_save / cache_every
            est_time = self.s_to_clock(size * time_per_sample)
            super(InsufficientSamplesError, self).__init__(
                "{} samples need to be generated (estimated time {})".format(size, est_time)
            )
        except FileNotFoundError:
            # no times have been recorded yet
            super(InsufficientSamplesError, self).__init__(
                "{} samples need to be generated;"
                " generate a small number first to estimate time".format(size)
            )

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


class PreprocessedData:
    """Represents a single list of data samples, preprocessed by a specific preprocessor."""

    def __init__(self, path, preprocessor, verbose=False):
        """Store the path to the data.

        Args:
            path (str): path to the directory containing pickle files with the processed data.
            preprocessor (class): preprocessor for this set.
            verbose (bool): whether to print debug statements.
        """
        self.path = path
        self.preprocessor = preprocessor
        self.name = os.path.basename(os.path.normpath(self.path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new PreprocessedData object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._size = None

    @property
    def size(self):
        """The size of the preprocessed set of data.

        Returns:
            (int)
        """
        if self._size is not None:
            return self._size

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
        try:
            self._size = utils.count_samples(self.path, self.verbose)
        except FileNotFoundError:
            os.makedirs(self.path, exist_ok=True)
            self._size = 0

        # store it in the metadata file
        utils.v_print(self.verbose, "writing size '{}' to '{}'".format(self._size, metadata_path))
        with open(metadata_path, "a+") as outfile:
            outfile.write("size: {}\n".format(self._size))

        return self._size

    def get(self, size=None):
        """Return `size` samples from this preprocessed dataset.

        Warning: this data comes in order. You'll probably want to shuffle it.

        Args:
            size (int): number of samples to return. If None, return all of them.
        Returns:
            tuple(list, list): samples.
        """
        if size is None:
            # get all the samples
            return utils.get_samples(self.path, self.size, self.verbose)

        max_size = self.size
        if max_size < size:
            raise InsufficientSamplesError(
                size - max_size, os.path.dirname(os.path.dirname(self.path)), self.verbose
            )

        return utils.get_samples(self.path, size, self.verbose)


class Group:
    """Represents a group of data samples, and contains references to all contained
    PreprocessedData."""

    def __init__(self, path, preprocessors, verbose=False):
        """Store the path to the group, to allow access to raw data.

        Args:
            path (str): path to group; should contain pickle files for the unprocessed set and
                        directories for each preprocessed set.
            preprocessors (list(class)): preprocessors for this dataset.
            verbose (bool): whether to print debug statements.
        """
        self.path = path
        self.preprocessors = preprocessors
        self.name = os.path.basename(os.path.normpath(path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new Group object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._unprocessed_size = None
        self._iterator = None

    @property
    def unprocessed_size(self):
        """The size of the unprocessed set of data.

        Returns:
            (int)
        """
        if self._unprocessed_size is not None:
            return self._unprocessed_size

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

    def get(self, size=None):
        """Get `size` samples from the unprocessed set.

        Warning: this data comes in order. You'll probably want to shuffle it.

        Args:
            size (int): number of samples to retrieve. If None, return all the samples.
        Returns:
            tuple(list, list): samples.
        """
        if size is None:
            # return all the samples
            return utils.get_samples(self.path, self.unprocessed_size, self.verbose)

        u_size = self.unprocessed_size
        if u_size < size:
            raise InsufficientSamplesError(size - u_size, os.path.dirname(self.path), self.verbose)

        return utils.get_samples(self.path, size, self.verbose)

    def assert_preprocessed_size(self, name, size):
        """Ensure that at least `size` samples have been processed with `func`.

        The preprocessor must accept a list of data points and a list of corresponding labels. It
        should return preprocessed versions of both data and label.

        Args:
            func (str): name of preprocessor to be applied to data.
            size (int): number of samples to preprocess.
        Raises:
            (utils.InsufficientSamplesError): there are not enough unprocessed samples.
        """
        if self.unprocessed_size < size:
            raise InsufficientSamplesError(size - self.unprocessed_size, self.path, self.verbose)

        # get unprocessed data
        unproc_files = sorted(file for file in os.listdir(self.path) if file.endswith(".pkl"))

        # make sure the PreprocessedData exists, by trying to get it
        proc_set = self[name]

        # get the current size of the PreprocessedData and the individual files which
        # have already been preprocessed
        count = proc_set.size
        current_proc_files = set(os.listdir(proc_set.path))

        preprocessor = proc_set.preprocessor

        # preprocess files, skipping those that are already preprocessed
        for file in unproc_files:
            if count >= size:
                break
            if file in current_proc_files:
                continue

            X, y = utils.from_pickle(os.path.join(self.path, file), self.verbose)
            count += len(X)

            # preprocess it!
            X, y = preprocessor.call(X, y)

            utils.to_pickle((X, y), os.path.join(proc_set.path, file))

        # invalidate the metadata file
        try:
            os.remove(os.path.join(proc_set.path, ".metadata"))
        except FileNotFoundError:
            pass

    def __iter__(self):
        self._iterator = iter(self.preprocessors)
        return self

    def _make_preprocessed_data(self, name, preproc):
        """Create a new PreprocessedData object from the given name and preprocessor.

        Args:
            name (str): name of preprocessor.
            preproc (class): preprocessor defined in datasets.py.
        Returns:
            (PreprocessedData)
        """
        return PreprocessedData(os.path.join(self.path, name), preproc, self.verbose)

    def __next__(self):
        """Iterate over the PreprocessedData for this Group.

        Returns:
            (PreprocessedData): the next preprocessed data that has been created under this group.
        """
        preproc = next(self._iterator)
        return self._make_preprocessed_data(preproc.__name__, preproc)

    def __getitem__(self, idx):
        """Allow access by name of PreprocessedData."""
        if isinstance(idx, str):
            for preproc in self.preprocessors:
                if preproc.__name__ == idx:
                    return self._make_preprocessed_data(idx, preproc)
            raise IndexError("preprocessor '{}' not found for this dataset".format(idx))

        if utils.is_preprocessor(idx):
            if idx in self.preprocessors:
                return self._make_preprocessed_data(idx.__name__, idx)

        raise IndexError(
            "Preprocessed set access permitted by str or preprocessor; got {}".format(type(idx))
        )


class ParamSet:
    """Represents a parameter set with all its datasets beneath it."""

    def __init__(self, path, preprocessors, verbose=False):
        """Store the path to the param set to allow access to child Groups and raw data.

        Args:
            path (str): path to parameter set; should be a subdirectory of a dataset directory.
            preprocessors (list(class)): preprocessors belonging to this dataset.
            verbose (bool): whether to print debug statements.
        """
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.preprocessors = preprocessors
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
                        raw_size = int(line.split(": ")[1])
                        return raw_size

        # count the number by hand
        raw_size = utils.count_samples(os.path.join(self.path, "raw"), self.verbose)

        # store it in the metadata file
        utils.v_print(self.verbose, "writing size '{}' to '{}'".format(raw_size, raw_metadata_path))
        with open(raw_metadata_path, "w+") as outfile:
            outfile.write("size: {}\n".format(raw_size))

        return raw_size

    @property
    def time_per_save(self):
        """Two standard deviations above the mean amount of time it takes to generate `cache_every`
        samples.
        """
        times = []
        for file in os.listdir(os.path.join(self.path, "raw", ".times")):
            delete = True
            with open(os.path.join(self.path, "raw", ".times", file), "r") as timefile:
                for line in timefile:
                    delete = False
                    times.append(float(line[:-1]))  # trailing newline

            if delete:  # remove empty files
                os.remove(os.path.join(self.path, "raw", ".times", file))

        if not times:
            raise FileNotFoundError("no timing information found")

        # variance requires at least two data points
        if len(times) < 2:
            utils.v_print(
                self.verbose, "only one timing data point found; estimated time may be inaccurate"
            )
            return times[0] * 1.1

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
            raise InsufficientSamplesError(size - self.raw_size, self.path)

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
        try:
            self._iterator = iter(
                {g for g in os.listdir(self.path) if not g.startswith(".") and g != "raw"}
            )
        except FileNotFoundError:
            self._iterator = iter(())

        return self

    def _make_group(self, name):
        """Create a new Group object by appending the name to self.path.

        Args:
            name (str): name of group.
        Returns:
            (Group)
        """
        return Group(os.path.join(self.path, name), self.preprocessors, self.verbose)

    def __next__(self):
        """Iterate over the Groups for this ParamSet.

        Returns:
            (Group): the next group that has been created under this parameter set.
        """
        return self._make_group(next(self._iterator))

    def __getitem__(self, idx):
        """Allow access by name of Group."""
        if not isinstance(idx, str):
            raise IndexError("Group access is only permitted by str; got {}".format(type(idx)))

        return self._make_group(idx)

    def __contains__(self, item):
        return not item.startswith(".") and item != "raw" and item in os.listdir(self.path)


class Dataset:
    """Represents a dataset with all its parameter sets beneath it."""

    def __init__(self, path, dataset, verbose=False):
        """Store the path to the dataset to allow access to child ParamSets.

        Args:
            path (str): path to dataset; should be a subdirectory of the cache directory.
            dataset (class): dataset decorated with @slurm_gen.dataset.
            verbose (bool): whether to print debug statements.
        """
        self.path = path
        self.dataset = dataset
        self.preprocessors = utils.get_preprocessors(dataset)
        self.name = os.path.basename(os.path.normpath(self.path))
        self.verbose = verbose
        utils.v_print(verbose, "creating new Dataset object with")
        utils.v_print(verbose, "  path '{}'".format(self.path))
        utils.v_print(verbose, "  and name '{}'".format(self.name))

        self._iterator = None

    def __iter__(self):
        try:
            self._iterator = iter(sorted(os.listdir(self.path)))
        except FileNotFoundError:
            self._iterator = iter(())

        return self

    def _make_ParamSet(self, param_str):
        """Create a new ParamSet object by appending the param_str to self.path.

        Args:
            param_str (str): a subdirectory of the dataset dir.
        Returns:
            (ParamSet)
        """
        return ParamSet(os.path.join(self.path, param_str), self.preprocessors, self.verbose)

    def __next__(self):
        """Iterate over the ParamSets for this dataset.

        Returns:
            (ParamSet): the next parameter set that has been generated for this dataset.
        """
        return self._make_ParamSet(next(self._iterator))

    def __getitem__(self, idx):
        """Access ParamSets by number, string, ParamObject, or dict of params."""
        elements = list(sorted(os.listdir(self.path)))

        if isinstance(idx, int):
            return self._make_ParamSet(elements[idx])

        if isinstance(idx, str):
            if idx not in elements:
                raise FileNotFoundError(
                    "no such parameter set found: {}".format(os.path.join(self.path, idx))
                )
            return self._make_ParamSet(idx)

        if isinstance(idx, utils.DefaultParamObject):
            return self._make_ParamSet(idx._to_string())

        if isinstance(idx, dict):
            idx = generator.paramClass(**dict)
            return self._make_ParamSet(idx._to_string())

        raise IndexError(
            "ParamSet access is only permitted with int or str; got {}".format(type(idx))
        )


class Cache:
    """Represents a cache location and all of its datasets."""

    def __init__(self, path, verbose=False):
        """Initialize the cache in this location.

        Args:
            path (str): path to the directory where datasets.py exists.
            verbose (bool): whether to print debug statements.
        """
        self.path = path
        self.cache_path = os.path.join(path, ".slurm_cache")
        os.makedirs(self.cache_path, exist_ok=True)
        self.verbose = verbose
        utils.v_print(verbose, "creating new Cache object with path '{}'".format(self.path))

        self._iterator = None
        self._datasets = None

    @property
    def datasets(self):
        """Get datasets defined in `datasets.py`.

        Returns:
            (list(class)): datasets.
        """
        if self._datasets is None:
            self._datasets = utils.get_datasets(self.path)
        return self._datasets

    def __iter__(self):
        self._iterator = iter(sorted(self.datasets))
        return self

    def _make_dataset(self, name, dataset):
        """Create a new Dataset object for the dataset by appending name to self.path.

        Args:
            name (str): name of dataset.
            dataset (class): dataset defined in datasets.py.
        Returns:
            (Dataset)
        """
        return Dataset(os.path.join(self.cache_path, name), dataset, self.verbose)

    def __next__(self):
        """Iterate over datasets defined in this directory.

        Returns:
            (Dataset): the next dataset in the cache.
        """
        dataset = next(self._iterator)
        return self._make_dataset(dataset.__name__, dataset)

    def __getitem__(self, idx):
        """Access either by name of Dataset or by index."""
        elements = list(sorted(self.datasets))

        if isinstance(idx, int):
            return self._make_dataset(datasets[idx].__name__, datasets[idx])

        if isinstance(idx, str):
            for dataset in elements:
                if dataset.__name__ == idx:
                    return self._make_dataset(idx, dataset)
            raise FileNotFoundError(
                "no dataset '{}' found in {}/datasets.py".format(idx, self.path)
            )

        if utils.is_dataset(idx):
            # the dataset itself
            return self._make_dataset(idx.__name__, idx)

        raise IndexError(
            "dataset access only permitted with int, str, or a generator; got {}".format(type(idx))
        )
