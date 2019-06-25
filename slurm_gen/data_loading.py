"""Data loading module for simulation data from pyMode.

Defines functions for getting data by either processing it or loading it from cache.

Kyle Roth. 2019-05-03.
"""


from glob import iglob as glob
import os
import pickle
from time import sleep
import warnings

from .utils import get_func_name, get_dataset_dir


class InsufficientSamplesError(Exception):
    """Helpful error messages for determining how much more time is needed to create the requested size of dataset."""
    def __init__(self, size, dataset_path):
        """Use the size needed to determine how much compute time it will take to generate that many samples.

        Args:
            size (int): number of samples needed.
            dataset_path (str): path to the dataset. Should include params but not set name ('train', 'test').
        """
        self.needed = size
        times = self._get_times(dataset_path)
        if times:
            est_time = self.s_to_clock(size * sum(times) / len(times))
            super(InsufficientSamplesError, self).__init__(
                '{} samples need to be generated (estimated time {})'.format(size, est_time))
        else:
            # no times have been recorded yet
            super(InsufficientSamplesError, self).__init__(
                '{} samples need to be generated; generate a small number first to estimate time'.format(size))

    @staticmethod
    def _get_times(dp):
        """Get the recorded times for previous data generation.

        Also compile the times into a single file so it's faster next time.

        Args:
            dp (str): path to the dataset, including params but not the set name.
        Returns:
            list(float): the collected times, in seconds.
        """
        times = []
        os.makedirs(os.path.join(dp, '.times'), exist_ok=True)
        for fp in glob(os.path.join(dp, '.times', '*.time')):
            with open(fp, 'r') as infile:
                times.extend(infile.read().strip().split())
            os.remove(fp)

        # store in a single file
        with open(os.path.join(dp, '.times', 'compiled.time'), 'w') as outfile:
            outfile.writelines([time + '\n' for time in times])

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
        return '{}:{}:{}'.format(str(h).zfill(2), str(m).zfill(2), str(s).zfill(2))


class CacheMutex:
    """Places and manages a lock file on data in a raw directory, so that concurrent threads don't assume access to the
    same pickle files."""
    def __init__(self, locked_path):
        """Ensure the path is available for locking, and lock it.

        Args:
            locked_path (str): path to directory to lock.
        """
        self.mutex_path = os.path.join(locked_path, 'cachemut.ex')

        # wait for mutex to be available
        while os.path.isfile(self.mutex_path):
            sleep(1)

        # lock the mutex
        os.makedirs(os.path.dirname(self.mutex_path), exist_ok=True)
        open(self.mutex_path, 'w').close()

    def __enter__(self):
        """Allow use with the Python `with` statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the mutex."""
        os.remove(self.mutex_path)


def _assign_data(size, target_path):
    """Move the specified number of unassigned samples into the dataset.

    For each dataset with a specific set of parameters, the unassigned samples appear in the 'raw' directory.

    Args:
        size (int): number of samples to include.
        target_path (str): specifies the location of the unprocessed dataset.
    Returns:
        (list): the features from the dataset.
        (list): the labels from the dataset.
    Raises:
        InsufficientSamplesError: if not enough unassigned samples are available.
    """
    raw_path = os.path.join(os.path.dirname(target_path), 'raw')

    # iterate through the files in the raw directory until enough samples have been collected
    with CacheMutex(raw_path):
        raw_files = iter(os.listdir(raw_path))
        raw_X = []
        raw_y = []
        to_delete = set()
        to_save = False
        try:
            while len(raw_X) < size:
                # get the next raw file
                raw_file = os.path.join(raw_path, next(raw_files))

                # don't try to read the mutex file as a pickle
                if raw_file.endswith('cachemut.ex'):
                    continue

                # load it
                try:
                    with open(raw_file, 'rb') as infile:
                        temp_X, temp_y = pickle.load(infile)
                except EOFError:
                    print(raw_file)
                    raise

                enough = size - len(raw_X)
                if len(temp_X) <= enough:
                    # add the whole thing and mark for deletion
                    raw_X.extend(temp_X)
                    raw_y.extend(temp_y)
                    to_delete.add(raw_file)
                else:
                    # only take the necessary samples
                    raw_X.extend(temp_X[:enough])
                    raw_y.extend(temp_y[:enough])
                    # save the rest
                    to_save = (raw_file, temp_X[enough:], temp_y[enough:])
        except StopIteration:
            # we ran out of files to load from. Good thing we didn't delete those files yet!
            print(raw_path)
            raise InsufficientSamplesError(size - len(raw_X), os.path.dirname(raw_path))

    # add collected samples to target set
    if os.path.isfile(os.path.join(target_path, 'none.pkl')):
        with open(os.path.join(target_path, 'none.pkl'), 'rb') as infile:
            X, y = pickle.load(infile)
        X.extend(raw_X[:size])
        y.extend(raw_y[:size])
    else:
        X = raw_X
        y = raw_y

    # cache unprocessed dataset
    with open(os.path.join(target_path, 'none.pkl'), 'wb') as outfile:
        pickle.dump((X, y), outfile)

    # delete marked files
    for raw_file in to_delete:
        os.remove(raw_file)

    # if present, save leftovers
    if to_save:
        raw_file, temp_X_left, temp_y_left = to_save
        with open(raw_file, 'wb') as outfile:
            pickle.dump((temp_X_left, temp_y_left), outfile)

    return X, y


def _load_dataset(size, unprocessed_path):
    """Load the unprocessed dataset from cache, including unassigned samples into the dataset if necessary.

    Args:
        size (int or NoneType): number of samples to include. If none, every sample is included.
        unprocessed_path (str): specifies the location of the unprocessed dataset.
    Returns:
        (list): the features from the dataset.
        (list): the labels from the dataset.
    """
    # try to load unprocessed dataset from cache
    if os.path.isfile(unprocessed_path):  # unprocessed cache exists
        with open(unprocessed_path, 'rb') as infile:
            X, y = pickle.load(infile)
        if size is None:
            return X, y
        if len(X) >= size:
            return X[:size], y[:size]

        # move unassigned samples into the dataset, and return the dataset
        return _assign_data(size - len(X), os.path.dirname(unprocessed_path))

    # if size is None, give them what we've got: nothing!
    if size is None:
        return [], []

    # no dataset created yet
    return _assign_data(size, os.path.dirname(unprocessed_path))


def _apply_preproc(preproc, batch_preproc, fn_name, X, y):
    """If available, apply the preprocessor to the dataset.

    Args:
        preproc (callable): Preprocessor function, with the call signature specified in the docstring for get_data.
        batch_preproc (bool): Whether to apply the preprocessor to the entire dataset or one point at a time.
        fn_name (str): Name of the preprocessor function, as given by inspect.getmembers.
        X (iterable): points in the dataset.
        y (iterable): labels in the dataset.
    Returns:
        (list): preprocessed points in the dataset.
        (list): preprocessed labels in the dataset.
    """
    if preproc:
        print('Applying preprocessor "{}".'.format(fn_name))
        if batch_preproc:
            X, y = preproc(X, y)
        else:
            try:
                X, y = zip(*(preproc(x, wai) for x, wai in zip(X, y)))
            except TypeError as e:
                if str(e).startswith('zip argument #') and str(e).endswith(' must support iteration'):
                    raise TypeError('preprocessor did not return the correct signature')
                raise
        print(' done.')
    return list(X), list(y)


def get_data(dataset, subset, size=None, params=None, preproc=None, batch_preproc=True, redo_preproc=False):
    """Return the train, val, or test set from the specified dataset, preprocessed as requested.

    If batch_preproc is set, the preprocessor must accept a list of data points and a list of corresponding labels.
    Otherwise, it must accept a single data point and its corresponding label. In either case, it should return
    preprocessed versions of both data and label.

    Args:
        dataset (str): name of dataset to load. Must be the name of a function defined in datasets.py.
        subset (str): specifies the requested dataset; one of {'train', 'val', 'train+val', 'test', 'toy'}.
        size (int or NoneType): number of samples to include. If None, returns the entire set of generated data.
        params (dict): parameters passed to data generation function.
        preproc (callable): preprocessing function to be applied to data. If batch_preproc is False, call signature must
                            allow (x, y) where x is a single data point and y is a label. If batch_preproc is True,
                            preproc is called on X, y where X is an iterable of all data points and y is an iterable of
                            all labels.
        batch_preproc (bool): if True, preproc is called on the entire dataset at once. Otherwise, preproc is called on
                              a single data point and label at a time.
        redo_preproc (bool): if True, preproc is called on the dataset whether or not a cached, preprocessed version has
                             already been created.
    Returns:
        list: data points, preprocessed as specified.
        list: labels corresponding to the data points.
    Raises:
        InsufficientSamplesError: if there are insufficient samples to create the requested dataset.
    """
    # default params to empty dictionary
    if params is None:
        params = dict()

    if subset.lower() == 'test':
        warnings.warn('getting test dataset; do not use except for producing final results')

    if subset.lower() == 'train+val':
        # use get_data twice, and combine output. If either call to get_data ends up insufficient, raise the sum as a
        # InsufficientSamplesError.
        needed_samples = size
        try:
            X, y = get_data(dataset, 'train', needed_samples, params, preproc, batch_preproc)
        except InsufficientSamplesError as e:
            needed_samples = e.needed
            X, y = get_data(dataset, 'train', size - needed_samples, params, preproc, batch_preproc)
        X2, y2 = get_data(dataset, 'val', needed_samples, params, preproc, batch_preproc)

        X.extend(X2)
        y.extend(y2)
        return X, y

    # get the name of the preprocessing function
    fn_name = get_func_name(preproc)

    # create names for the dataset and the unprocessed version
    dataset_dir = get_dataset_dir(dataset, params)
    pickle_path = os.path.join(dataset_dir, subset.lower(), '{}.pkl'.format(fn_name))

    # ensure the directory for this set is available
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    # try to load data from cache
    if not pickle_path.endswith('none.pkl') and os.path.isfile(pickle_path) and not redo_preproc:
        # preprocessed cache exists and we want to use it
        with open(pickle_path, 'rb') as infile:
            X, y = pickle.load(infile)
            if size is None:
                return X, y
            if len(X) >= size:
                # if it's enough, return it
                return X[:size], y[:size]
    else:
        # there wasn't enough preprocessed data in the cache (or we aren't getting preprocessed data)
        # so create empty containers for when we get the rest
        X = []
        y = []

    # get the unprocessed data either from cache or using _create_data
    unprocessed_path = os.path.join(os.path.dirname(pickle_path), 'none.pkl')  # location of unprocessed dataset
    unproc_X, unproc_y = _load_dataset(size, unprocessed_path)

    # X and y should be the beginning of unproc_X and unproc_y, so only preprocess the end
    unproc_X = unproc_X[len(X):]
    unproc_y = unproc_y[len(y):]

    # apply preprocessor
    unproc_X, unproc_y = _apply_preproc(preproc, batch_preproc, fn_name, unproc_X, unproc_y)

    # add the newly preprocessed data to whatever we had before
    X.extend(unproc_X)
    y.extend(unproc_y)

    # cache the preprocessed dataset
    if not pickle_path.endswith('none.pkl'):
        with open(pickle_path, 'wb') as outfile:
            pickle.dump((X, y), outfile)

    return X, y
