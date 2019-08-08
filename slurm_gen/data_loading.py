"""Data loading module for simulation data from pyMode.

Defines functions for getting data by either processing it or loading it from cache.

Kyle Roth. 2019-05-03.
"""


from glob import iglob as glob
import os
import warnings

from slurm_gen import datasets
from slurm_gen import utils


class InsufficientSamplesError(Exception):
    """Helpful error messages for determining how much more time is needed to create the requested size of dataset."""
    def __init__(self, size, dataset_path, verbose=False):
        """Use the size needed to determine how much compute time it will take to generate that many samples.

        Args:
            size (int): number of samples needed.
            dataset_path (str): path to the dataset. Should include params but not set name ('train', 'test').
            verbose (bool): print debugging statements to stdout. Useful for development.
        """
        self.needed = size
        times = self._get_times(dataset_path, verbose)
        if times:
            est_time = self.s_to_clock(size * sum(times) / len(times))
            super(InsufficientSamplesError, self).__init__(
                '{} samples need to be generated (estimated time {})'.format(size, est_time))
        else:
            # no times have been recorded yet
            super(InsufficientSamplesError, self).__init__(
                '{} samples need to be generated; generate a small number first to estimate time'.format(size))

    @staticmethod
    def _get_times(dp, verbose=False):
        """Get the recorded times for previous data generation.

        Also compile the times into a single file so it's faster next time.

        Args:
            dp (str): path to the dataset, including params but not the set name.
            verbose (bool): print debugging statements to stdout. Useful for development.
        Returns:
            list(float): the collected times, in seconds.
        """
        times = []
        os.makedirs(os.path.join(dp, '.times'), exist_ok=True)
        for fp in glob(os.path.join(dp, '.times', '*.time')):
            with open(fp, 'r') as infile:
                times.extend(infile.read().strip().split())
            os.remove(fp)
        utils.v_print(verbose, 'Collected {} timings from "{}".'.format(len(times), os.path.join(dp, '.times')))

        # store in a single file
        with open(os.path.join(dp, '.times', 'compiled.time'), 'w') as outfile:
            outfile.writelines([time + '\n' for time in times])
        utils.v_print(verbose, 'Wrote timings to "{}"'.format(os.path.join(dp, '.times', 'compiled.time')))

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


def _collect_raw(raw_path, raw_files, size, verbose=False):
    """Collect raw samples to be assigned by _assign_data.

    Instead of deleting the raw files, return the files marked for deletion. This allows _assign_data to create the
    updated cache before deletion, avoiding the risk of the process being killed after deletion but before writing the
    updated cache. Yay for not losing your data!

    Args:
        raw_path (str): path to location of raw files.
        raw_files (list(str)): list of raw files containing samples to collect.
        size (int): number of samples to collect.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): features for raw samples collected from files.
        (list): labels for raw samples collected from files.
        (set(str)): set of raw files to be deleted.
        (bool or tuple(str, list, list)): either False (meaning no leftovers need to be saved), or a tuple with:
                                          - the path to the raw file where leftovers should be written,
                                          - the raw features to be written,
                                          - the raw labels to be written.
    """
    raw_files = iter(raw_files)
    raw_X = []
    raw_y = []
    to_delete = set()
    to_save = False

    try:
        while len(raw_X) < size:
            utils.v_print(verbose, 'Collected {}/{} samples so far.'.format(len(raw_X), size))
            # get the next raw file
            raw_file = os.path.join(raw_path, next(raw_files))

            # don't try to read the mutex file as a pickle
            if raw_file.endswith('cachemut.ex'):
                continue

            utils.v_print(verbose, 'Reading raw file "{}".'.format(raw_file))

            # load it
            temp_X, temp_y = utils.from_pickle(raw_file, verbose)

            # skip if loading was unsuccessful
            if temp_X == [] and temp_y == []:
                continue

            enough = size - len(raw_X)
            if len(temp_X) <= enough:
                # add the whole thing and mark for deletion
                raw_X.extend(temp_X)
                raw_y.extend(temp_y)
                to_delete.add(raw_file)
                utils.v_print(
                    verbose, 'Collected {} samples from raw file. Still need {} more.'.format(
                        len(temp_X), enough - len(temp_X)))
            else:
                # only take the necessary samples
                raw_X.extend(temp_X[:enough])
                raw_y.extend(temp_y[:enough])
                # save the rest
                to_save = (raw_file, temp_X[enough:], temp_y[enough:])
                utils.v_print(
                    verbose, 'Collected {} samples from raw file, with {} left over.'.format(
                        enough, len(to_save[1])))
    except StopIteration:
        # we ran out of files to load from. Good thing we didn't delete those files yet!
        utils.v_print(verbose, 'Ran out of raw files before enough samples were found.')
        raise InsufficientSamplesError(size - len(raw_X), os.path.dirname(raw_path))

    return raw_X, raw_y, to_delete, to_save


def _assign_data(size, target_path, verbose=False):
    """Move the specified number of unassigned samples into the dataset.

    For each dataset with a specific set of parameters, the unassigned samples appear in the 'raw' directory.

    Args:
        size (int): number of samples to include.
        target_path (str): specifies the location of the unprocessed dataset.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): the features from the dataset.
        (list): the labels from the dataset.
    Raises:
        InsufficientSamplesError: if not enough unassigned samples are available.
    """
    raw_path = os.path.join(os.path.dirname(target_path), 'raw')
    utils.v_print(verbose, 'Raw path: "{}"'.format(raw_path))

    # iterate through the files in the raw directory until enough samples have been collected
    utils.v_print(verbose, 'Mutex obtained for raw path.')

    raw_files = os.listdir(raw_path)
    utils.v_print(verbose, '{} raw files found.'.format(len(raw_files)))

    # collect samples from the raw files, and raise an InsufficientSamplesError if there aren't enough samples
    raw_X, raw_y, to_delete, to_save = _collect_raw(raw_path, raw_files, size, verbose)

    # add collected samples to target set
    utils.v_print(verbose, 'Target path: "{}".'.format(target_path))
    if os.path.isfile(os.path.join(target_path, 'none.pkl')):
        X, y = utils.from_pickle(os.path.join(target_path, 'none.pkl'), verbose)
        utils.v_print(verbose, 'Collected {} samples from target cache.'.format(len(X)))

        X.extend(raw_X[:size])
        y.extend(raw_y[:size])
    else:
        X = raw_X
        y = raw_y
    utils.v_print(
        verbose, 'Added {} samples to existing set of {} samples, for a total of {}.'.format(
            len(raw_X), len(X) - len(raw_X), len(X)))

    # cache unprocessed dataset
    utils.v_print(verbose, 'Caching {} samples to target path.'.format(len(X)))
    utils.to_pickle((X, y), os.path.join(target_path, 'none.pkl'))

    # we saved deletions and writing of leftovers until the assigned data was cached,
    # just in case this process were killed

    # delete marked files
    for raw_file in to_delete:
        utils.v_print(verbose, 'Deleting raw file: "{}"'.format(raw_file))
        os.remove(raw_file)

    # if present, save leftovers
    if to_save:
        raw_file, temp_X_left, temp_y_left = to_save
        utils.v_print(verbose, 'Saving {} leftover samples to "{}".'.format(len(temp_X_left), raw_file))
        utils.to_pickle((temp_X_left, temp_y_left), raw_file)


    utils.v_print(verbose, 'Returning {} samples.'.format(len(X)))
    return X, y


def _load_unprocessed_dataset(size, unprocessed_path, verbose=False):
    """Load the unprocessed dataset from cache, including unassigned samples into the dataset if necessary.

    Args:
        size (int or NoneType): number of samples to include. If none, every sample is included.
        unprocessed_path (str): specifies the location of the unprocessed dataset.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): the features from the dataset.
        (list): the labels from the dataset.
    """
    # try to load unprocessed dataset from cache
    if os.path.isfile(unprocessed_path):  # unprocessed cache exists
        utils.v_print(verbose, 'Unprocessed cache found: "{}"'.format(unprocessed_path))
        X, y = utils.from_pickle(unprocessed_path, verbose)
        if size is None:
            utils.v_print(verbose, 'No size specified; returning (len(X), len(y)) == ({}, {}).'.format(len(X), len(y)))
            return X, y
        if len(X) >= size:
            utils.v_print(
                verbose,
                'Size == {size} specified; returning first {size} of (len(X), len(y)) == ({x}, {y}).'.format(
                    size=size,
                    x=len(X),
                    y=len(y)
                )
            )
            return X[:size], y[:size]

        # move unassigned samples into the dataset, and return the dataset
        utils.v_print(verbose, 'Insufficient data ({} < {}). Calling _assign_data.'.format(len(X), size))
        return _assign_data(size - len(X), os.path.dirname(unprocessed_path), verbose)

    # if size is None, give them what we've got: nothing!
    if size is None:
        utils.v_print(verbose, 'No size specified and no unprocessed cache found. Returning empty lists.')
        return [], []

    # no dataset created yet
    utils.v_print(verbose, 'No unprocessed cache. Calling _assign_data.')
    return _assign_data(size, os.path.dirname(unprocessed_path), verbose)


def _apply_preproc(preproc, batch_preproc, X, y, verbose=False):
    """If available, apply the preprocessor to the dataset.

    Args:
        preproc (callable): Preprocessor function, with the call signature specified in the docstring for get_data.
        batch_preproc (bool): Whether to apply the preprocessor to the entire dataset or one point at a time.
        X (iterable): points in the dataset.
        y (iterable): labels in the dataset.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): preprocessed points in the dataset.
        (list): preprocessed labels in the dataset.
    """
    if preproc:
        utils.v_print(verbose, 'Applying preprocessor "{}".'.format(preproc.__name__))
        if batch_preproc:
            utils.v_print(verbose, '    (using batch application)')
            X, y = preproc(X, y)
        else:
            utils.v_print(verbose, '    (using individual application)')
            try:
                X, y = zip(*(preproc(x, wai) for x, wai in zip(X, y)))
            except TypeError as e:
                if str(e).startswith('zip argument #') and str(e).endswith(' must support iteration'):
                    raise TypeError('preprocessor did not return the correct signature')
                raise
    else:
        utils.v_print(verbose, 'Preprocessor not run since none given.')
    return list(X), list(y)


def _load_processed_dataset(pickle_path, size, preproc, batch_preproc, redo_preproc, verbose=False):
    """Load and return the processed dataset, processing more data if necessary.

    Args:
        pickle_path (str): path to the target pickle file. Should be dataset_dir plus the set name (e.g. 'train') plus
                           the preprocessor name.
        size (int): number of samples to get.
        preproc (function): preprocessing function to apply to data.
        batch_preproc (bool): if True, preproc is called on the entire dataset at once. Otherwise, preproc is called on
                              a single data point and label at a time.
        redo_preproc (bool): if True, preproc is called on the dataset whether or not a cached, preprocessed version has
                             already been created.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): data points, preprocessed as specified.
        (list): labels corresponding to the data points.
    """
    # try to load data from cache
    if not pickle_path.endswith('none.pkl') and os.path.isfile(pickle_path) and not redo_preproc:
        # preprocessed cache exists and we want to use it
        utils.v_print(verbose, 'Preprocessed cache exists.')
        X, y = utils.from_pickle(pickle_path, verbose)
        if size is None:
            utils.v_print(verbose, 'No size specified; returning (len(X), len(y)) == ({}, {}).'.format(len(X), len(y)))
            return X, y
        if len(X) >= size:
            # if it's enough, return it
            utils.v_print(
                verbose,
                'Size == {size} specified; returning first {size} of (len(X), len(y)) == ({x}, {y}).'.format(
                    size=size,
                    x=len(X),
                    y=len(y)
                )
            )
            return X[:size], y[:size]
        utils.v_print(verbose, 'Getting unprocessed dataset.')
        utils.v_print(verbose, '    (cache size insufficient ({} < {}))'.format(len(X), size))
    else:
        # there wasn't enough preprocessed data in the cache (or we aren't getting preprocessed data)
        # so create empty containers for when we get the rest
        utils.v_print(
            verbose and (pickle_path.endswith('none.pkl') or not os.path.isfile(pickle_path)),
            'Getting unprocessed dataset.'
        )
        utils.v_print(verbose and redo_preproc, '    (redo_preproc set to True)')
        X = []
        y = []

    # get the unprocessed data either from cache or using _create_data
    unprocessed_path = os.path.join(os.path.dirname(pickle_path), 'none.pkl')  # location of unprocessed dataset
    utils.v_print(verbose, 'Unprocessed path: "{}"'.format(unprocessed_path))
    unproc_X, unproc_y = _load_unprocessed_dataset(size, unprocessed_path, verbose)

    # X and y should be the beginning of unproc_X and unproc_y, so only preprocess the end
    utils.v_print(verbose and X, 'Cutting first {} from the unprocessed dataset'.format(len(X)))
    utils.v_print(verbose and X, '    (len cut from {} to {})'.format(len(unproc_X), len(unproc_X) - len(X)))
    utils.v_print(verbose and X, '    (first {} already preprocessed)'.format(len(X)))
    unproc_X = unproc_X[len(X):]
    unproc_y = unproc_y[len(y):]

    # apply preprocessor
    unproc_X, unproc_y = _apply_preproc(preproc, batch_preproc, unproc_X, unproc_y, verbose)

    # add the newly preprocessed data to whatever we had before
    utils.v_print(verbose, 'Adding {} newly-preprocessed samples to set.'.format(len(unproc_X)))
    X.extend(unproc_X)
    y.extend(unproc_y)

    # cache the preprocessed dataset
    if not pickle_path.endswith('none.pkl'):
        utils.v_print(verbose, 'Writing resulting preprocessed samples to cache: "{}".'.format(pickle_path))
        utils.to_pickle((X, y), pickle_path)

    return X, y


def get_data(dataset, subset, size=None, params=None, preproc=None, batch_preproc=True, redo_preproc=False,
             verbose=False):
    """Return the train, val, or test set from the specified dataset, preprocessed as requested.

    If batch_preproc is set, the preprocessor must accept a list of data points and a list of corresponding labels.
    Otherwise, it must accept a single data point and its corresponding label. In either case, it should return
    preprocessed versions of both data and label.

    Args:
        dataset (str): name of dataset to load. Must be the name of a function defined in datasets.py.
        subset (str): specifies the requested dataset; one of {'train', 'val', 'train+val', 'test', 'toy'}.
        size (int or NoneType): number of samples to include. If None, returns the entire set of generated data.
        params (dict or subclass of utils.DefaultParamObject): parameters passed to data generation function. If params
                                                               is a dict, it will be converted to the param class
                                                               corresponding to the dataset.
        preproc (callable): preprocessing function to be applied to data. If batch_preproc is False, call signature must
                            allow (x, y) where x is a single data point and y is a label. If batch_preproc is True,
                            preproc is called on X, y where X is an iterable of all data points and y is an iterable of
                            all labels.
        batch_preproc (bool): if True, preproc is called on the entire dataset at once. Otherwise, preproc is called on
                              a single data point and label at a time.
        redo_preproc (bool): if True, preproc is called on the dataset whether or not a cached, preprocessed version has
                             already been created.
        verbose (bool): print debugging statements to stdout. Useful for development.
    Returns:
        (list): data points, preprocessed as specified.
        (list): labels corresponding to the data points.
    Raises:
        InsufficientSamplesError: if there are insufficient samples to create the requested dataset.
    """
    # turn the params dict into the param object corresponding to this dataset
    if isinstance(params, dict):
        params = getattr(datasets, dataset).paramClass(**params)
    elif params is None:
        params = getattr(datasets, dataset).paramClass()

    if subset.lower() == 'test':
        warnings.warn('getting test dataset; do not use except for producing final results')
    elif subset.lower() == 'train+val':
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
    fn_name = utils.get_func_name(preproc)
    utils.v_print(verbose, 'Preprocessor name: "{}"'.format(fn_name))

    # create names for the dataset and the unprocessed version
    dataset_dir = utils.get_dataset_dir(dataset, params)
    utils.v_print(verbose, 'Dataset directory: "{}"'.format(dataset_dir))
    pickle_path = os.path.join(dataset_dir, subset.lower(), '{}.pkl'.format(fn_name))
    utils.v_print(verbose, 'Pickle path: "{}"'.format(pickle_path))

    # ensure the directory for this set is available
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    return _load_processed_dataset(pickle_path, size, preproc, batch_preproc, redo_preproc, verbose)
