"""Objects for representing and managing datasets in the cache.

The hierarchy is Cache -> Dataset -> ParamSet -> Group -> PreprocessedGroup

Kyle Roth. 2019-08-05.
"""


import os

from slurm_gen import utils


class PreprocessedData:
    """Represents a single list of data samples, preprocessed by a specific
    preprocessor."""

    def __init__(self, cache_path):
        """Store the path to the data.

        Args:
            cache_path (str): path to the pickle file containing the data.
        """
        self.path = cache_path
        self.name = ".".join(
            os.path.basename(os.path.normpath(cache_path)).split(".")[:-1]
        )

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
        metadata_path = os.path.join(
            os.path.dirname(os.path.normpath(self.path)), ".metadata"
        )
        if os.path.isfile(metadata_path):
            with open(metadata_path) as infile:
                for line in infile:
                    if line.startswith('size "{}"'.format(self.name)):
                        self._size = int(line.split(": ")[1])
                        return self._size

        # count the number by hand
        self._size = len(utils.from_pickle(self.path)[1])

        # store it in the metadata file
        with open(metadata_path, "a+") as outfile:
            outfile.write('size "{}": {}\n'.format(self.name, self._size))

        return self._size

    def __len__(self):
        return self.size


class Group:
    """Represents a group of data samples, and contains references to all contained
    PreprocessedData."""

    def __init__(self, cache_path):
        """Store the path to the group, to allow access to raw data.

        Args:
            cache_path (str): path to group; should contain one pickle file for each
                              preprocessed set.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self._iterator = None
        self._elements = {
            item for item in os.listdir(self.path) if not item.startswith(".")
        }

        self._unprocessed_size = None

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
            with open(metadata_path) as infile:
                for line in infile:
                    if line.startswith('size "none"'):
                        self._unprocessed_size = int(line.split(": ")[1])
                        return self._unprocessed_size

        # count the number by hand
        self._unprocessed_size = len(
            utils.from_pickle(os.path.join(self.path, "none.pkl"))[1]
        )

        # store it in the metadata file
        with open(metadata_path, "a+") as outfile:
            outfile.write('size "none": {}\n'.format(self._unprocessed_size))

        return self._unprocessed_size

    def __iter__(self):
        self._iterator = iter(self._elements)
        return self

    def __next__(self):
        """Iterate over the PreprocessedData for this Group.

        Returns:
            (PreprocessedData): the next preprocessed data that has been created under
                                this group.
        """
        return PreprocessedData(os.path.join(self.path, next(self._iterator)))


class ParamSet:
    """Represents a parameter set with all its datasets beneath it."""

    def __init__(self, cache_path):
        """Store the path to the param set to allow access to child Groups and raw data.

        Args:
            cache_path (str): path to parameter set; should be a subdirectory of a
                              dataset directory.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self._iterator = None
        self._elements = {
            item
            for item in os.listdir(self.path)
            if not item.startswith(".") and item != "raw"
        }

        self._raw_size = None

    @property
    def raw_size(self):
        """The size of the raw set of data.

        Returns:
            (int)
        """
        if self._raw_size is not None:
            return self._raw_size

        # try to read it from the metadata file
        raw_metadata_path = os.path.join(self.path, "raw", ".metadata")
        if os.path.isfile(raw_metadata_path):
            with open(raw_metadata_path) as infile:
                for line in infile:
                    if line.startswith("size: "):
                        self._raw_size = int(line.split(": ")[1])
                        return self._raw_size

        # count the number by hand
        self._raw_size = utils.count_samples(os.path.join(self.path, "raw"))

        # store it in the metadata file
        with open(raw_metadata_path, "w+") as outfile:
            outfile.write("size: {}\n".format(self._raw_size))

        return self._raw_size

    def __iter__(self):
        self._iterator = iter(self._elements)
        return self

    def __next__(self):
        """Iterate over the Groups for this ParamSet.

        Returns:
            (Group): the next group that has been created under this parameter set.
        """
        return Group(os.path.join(self.path, next(self._iterator)))

    def __getitem__(self, idx):
        """Allow access by name of Group."""
        if not isinstance(idx, str):
            raise IndexError(
                "Group access is only permitted by str; got {}".format(type(idx))
            )
        if idx not in self._elements:
            raise IndexError(
                "no such group found: {}".format(os.path.join(self.path, idx))
            )
        return Group(os.path.join(self.path, idx))


class Dataset:
    """Represents a dataset with all its parameter sets beneath it."""

    def __init__(self, cache_path):
        """Store the path to the dataset to allow access to child ParamSets.

        Args:
            cache_path (str): path to dataset; should be a subdirectory of the cache
                              directory.
        """
        self.path = cache_path
        self.name = os.path.basename(os.path.normpath(cache_path))
        self._iterator = None
        self._elements = list(sorted(os.listdir(self.path)))

    def __iter__(self):
        self._iterator = iter(self._elements)
        return self

    def __next__(self):
        """Iterate over the ParamSets for this dataset.

        Returns:
            (ParamSet): the next parameter set that has been generated for this dataset.
        """
        return ParamSet(os.path.join(self.path, next(self._iterator)))

    def __getitem__(self, idx):
        """Access either by name of ParamSet or by index."""
        if isinstance(idx, int):
            return ParamSet(os.path.join(self.path, self._elements[idx]))
        if isinstance(idx, str):
            if idx not in self._elements:
                raise ValueError(
                    "no such parameter set found: {}".format(
                        os.path.join(self.path, idx)
                    )
                )
            return ParamSet(os.path.join(self.path, idx))
        raise IndexError(
            "ParamSet access is only permitted with int or str; got {}".format(
                type(idx)
            )
        )


class Cache:
    """Represents a cache location and all of its datasets."""

    def __init__(self):
        """For now, the cache location is constant."""
        self.path = utils.get_cache_dir()
        self._iterator = None
        self._elements = list(
            sorted(item for item in os.listdir(self.path) if not item.startswith("."))
        )

    def __iter__(self):
        self._iterator = iter(self._elements)
        return self

    def __next__(self):
        """Iterate over datasets in the cache.

        Returns:
            (Dataset): the next dataset in the cache.
        """
        return Dataset(os.path.join(self.path, next(self._iterator)))

    def __getitem__(self, idx):
        """Access either by name of Dataset or by index."""
        if isinstance(idx, int):
            return Dataset(os.path.join(self.path, self._elements[idx]))
        if isinstance(idx, str):
            if idx not in self._elements:
                raise ValueError(
                    "no such dataset found: {}".format(os.path.join(self.path, idx))
                )
            return Dataset(os.path.join(self.path, idx))
        raise IndexError(
            "Dataset access is only permitted with int or str; got {}".format(type(idx))
        )
