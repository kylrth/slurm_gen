"""Objects for representing and managing datasets in the cache.

The hierarchy is Cache -> Dataset -> ParamSet -> Raw
                                              -> Group -> PreprocessedGroup

Kyle Roth. 2019-08-05.
"""


import os

from slurm_gen import utils


class Group:
    """Represents a group of data samples and all of their preprocessed versions."""
    def __init__(self, cache_path):
        """Store the path to the group, to allow access to raw data.

        Args:
            cache_path (str): path to group; should contain one pickle file for each preprocessed set.
        """
        self.path = cache_path


class ParamSet:
    """Represents a parameter set with all its datasets beneath it."""
    def __init__(self, cache_path):
        """Store the path to the param set to allow access to child Groups and raw data.

        Args:
            cache_path (str): path to parameter set; should be a subdirectory of a dataset directory.
        """
        self.path = cache_path
        self.iterator = None
        self.elements = {item for item in os.listdir(self.path) if not item.startswith('.') and item != 'raw'}


class Dataset:
    """Represents a dataset with all its parameter sets beneath it."""
    def __init__(self, cache_path):
        """Store the path to the dataset to allow access to child ParamSets.

        Args:
            cache_path (str): path to dataset; should be a subdirectory of the cache directory.
        """
        self.path = cache_path
        self.iterator = None
        self.elements = list(sorted(os.listdir(self.path)))

    def __iter__(self):
        self.iterator = iter(self.elements)
        return self

    def __next__(self):
        """Iterate over the ParamSets for this dataset.

        Returns:
            (ParamSet): the next parameter set that has been generated for this dataset.
        """
        return ParamSet(os.path.join(self.path, next(self.iterator)))

    def __getitem__(self, idx):
        """Access either by name of ParamSet or by index."""
        if isinstance(idx, int):
            return ParamSet(os.path.join(self.path, self.elements[idx]))
        if isinstance(idx, str):
            if idx not in self.elements:
                raise ValueError('no such parameter set found: {}'.format(os.path.join(self.path, idx)))
            return ParamSet(os.path.join(self.path, idx))
        raise IndexError('ParamSet access is only permitted with int or str; got {}'.format(type(idx)))


class Cache:
    """Represents a cache location and all of its datasets."""
    def __init__(self):
        """For now, the cache location is constant."""
        self.path = utils.get_cache_dir()
        self.iterator = None
        self.elements = list(sorted(os.listdir(self.path)))

    def __iter__(self):
        self.iterator = iter(self.elements)
        return self

    def __next__(self):
        """Iterate over datasets in the cache.

        Returns:
            (Dataset): the next dataset in the cache.
        """
        return Dataset(os.path.join(self.path, next(self.iterator)))

    def __getitem__(self, idx):
        """Access either by name of Dataset or by index."""
        if isinstance(idx, int):
            return Dataset(os.path.join(self.path, self.elements[idx]))
        if isinstance(idx, str):
            if idx not in self.elements:
                raise ValueError('no such dataset found: {}'.format(os.path.join(self.path, idx)))
            return Dataset(os.path.join(self.path, idx))
        raise IndexError('Dataset access is only permitted with int or str; got {}'.format(type(idx)))
