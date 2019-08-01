"""Command line interface for SLURM_gen.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen import utils


def get_count(path):
    """Get the quantity of data samples available at the dataset path.

    Looks for a metadata file named '.metadata', which should contain the size. If not, it collects the pickle files and
    determines the size.

    If the path is to a 'raw' set of samples, just the number of samples is returned. Otherwise, a dict mapping
    preprocessor names to numbers of samples is returned.

    Args:
        path (str): path to directory.
    Returns:
        (int or dict): number of samples.
    """
    # check to see if the .metadata file contains the size
    if os.path.isfile(os.path.join(path, '.metadata')):
        with open(os.path.join(path, '.metadata'), 'r') as metadata:
            if path.endswith('raw'):
                for line in metadata:
                    if line.startswith('size: '):
                        # for raw, there is no preprocessing, so only a number is desired
                        return int(line.split(': ')[1])
            else:
                out = {}
                for line in metadata:
                    if line.startswith('size "'):
                        out[line.split('"')[1]] = int(line.split(': ')[1])
                return out

    # count the samples in the pkl files by hand
    if path.endswith('raw'):
        count = 0
        for file in os.listdir(path):
            if file.endswith('.pkl'):
                count += len(utils.from_pickle(os.path.join(path, file))[1])

        # save the count for next time
        with open(os.path.join(path, '.metadata'), 'w+') as metadata:
            metadata.write('size: {}\n'.format(count))
    else:
        count = {}
        for file in os.listdir(path):
            if file.endswith('.pkl'):
                count[file[:-4]] = len(utils.from_pickle(os.path.join(path, file))[1])


    return count


def get_counts(dataset):
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
    Returns:
        (dict): a map from subset strings to counts, at the necessary depths.
    """
    dataset_dir = os.path.join(utils.get_cache_dir(), dataset)
    out = {}

    for params in os.listdir(dataset_dir):
        out[params] = {}
        for set_name in os.listdir(os.path.join(dataset_dir, params)):
            if not set_name.startswith('.'):  # don't grab things like '.times'
                out[params][set_name] = get_count(os.path.join(dataset_dir, params, set_name))

    return out


def _list():
    """List the datasets along with the number of samples generated for them."""
    counts = {}
    for dataset in os.listdir(utils.get_cache_dir()):
        if not dataset.startswith('.'):
            counts[dataset] = get_counts(dataset)

    name_col_len = max(len(d) for p in counts for d in counts[p])
    raw_col_len = max(len(str(counts[p][d]['raw'])) for p in counts for d in counts[p])
    train_col_len = max(len(str(counts[p][d]['train'])) for p in counts for d in counts[p])
    val_col_len = max(len(str(counts[p][d]['val'])) for p in counts for d in counts[p])
    test_col_len = max(len(str(counts[p][d]['test'])) for p in counts for d in counts[p])

    print('Name'.ljust(name_col_len + 4), 'Raw'.ljust(raw_col_len), 'Train'.ljust(train_col_len), sep='|', end='')
    print('Val'.ljust(val_col_len), 'Test'.ljust(test_col_len), sep='|')

    print(counts)


def _move(p):
    """Move unlabeled samples into a particular set."""
    # TODO: don't allow moving to 'train+val', but allow everything else
    # TODO: don't move out of 'test'


def main(p):
    """Use the parsed options to perform actions.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.command == 'list':
        _list()
    elif p.command == 'move':
        _move(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command line interface for SLURM_gen.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # required arguments
    parser.add_argument('command', help='command to perform. One of {"list"}')

    main(parser.parse_args())
