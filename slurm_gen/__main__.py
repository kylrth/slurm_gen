"""Command line interface for SLURM_gen.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen import utils


def print_sizes(params, counts):
    """Parameter strings are often very long, so we print them (and their counts) nicely.

    Args:
        params (str): parameters and their values, separated by pipe characters ('|').
        counts (dict): a map from set names to dicts of the quantities for each preprocessor used. The exception is
                       'raw', which should map to a single number since raw data is not preprocessed.
    """
    # collect strings of parameters two wide
    param_iter = iter(params.split('|'))
    param_strings = []
    while True:
        try:
            param_strings.append(next(param_iter) + '|')
            param_strings[-1] += next(param_iter) + '|'
        except StopIteration:
            break

    # collect strings of counts
    count_strings = []
    for set_name in counts:
        if set_name == 'raw':
            count_strings.append(' raw: {}'.format(counts[set_name]))
        else:
            count_strings.append(' {}:'.format(set_name))
            keys = iter(counts[set_name].keys())
            try:
                key = next(keys)
                count_strings[-1] += ' {}: {}'.format(key, counts[set_name][key])
                while True:
                    key = next(keys)
                    count_strings.append(' ' * (len(set_name) + 3) + '{}: {}'.format(key, counts[set_name][key]))
            except StopIteration:
                pass

    # print the param and count strings next to each other
    param_iter = iter(param_strings)
    count_iter = iter(count_strings)

    # right align the param strings, with 4 spaces before the whole thing
    param_spacing = 4 + max(len(string) for string in param_strings)

    while True:
        # print the next strings
        try:
            print(next(param_iter).rjust(param_spacing), end='')
        except StopIteration:
            # print the rest of the count strings, if any
            try:
                while True:
                    print(' ' * (param_spacing - 1) + '|' + next(count_iter))
            except StopIteration:
                break
        try:
            print(next(count_iter))
        except StopIteration:
            # print the rest of the param strings, if any
            try:
                while True:
                    print(next(param_iter).rjust(param_spacing))
            except StopIteration:
                break


def single_list(dataset, verbose=False):
    """Print the count information for a single dataset.

    Args:
        dataset (str): name (not path) of dataset.
        verbose (bool): whether to print debug statements.
    """
    counts = utils.get_counts(dataset, verbose)
    count = 1
    for params in counts:
        print('Param set #{}:'.format(count))
        print_sizes(params, counts[params])
        count += 1


def _list(p):
    """List the datasets along with the number of samples generated for them.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.dataset is None:
        # get all of them!
        divider = '-' * 80 + '\n'
        did_print = False

        sep = ''
        for dataset in os.listdir(utils.get_cache_dir()):
            if not dataset.startswith('.'):
                print(sep, end='')
                sep = divider
                print('{}:'.format(dataset.upper()))
                single_list(dataset, p.verbose)
                did_print = True

        if not did_print:
            print('No datasets found. Generate one!')
    else:
        # get just the one dataset
        try:
            single_list(p.dataset.lower(), p.verbose)
        except FileNotFoundError:
            print('No dataset found by name {}'.format(p.dataset.lower()))


def _move(p):
    """Move samples between sets, confirming the choice before doing so.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.to_set.lower() == 'train+val':
        raise ValueError('the name "train+val" is reserved for combining the "train" and "val" sets')
    if p.from_set.lower() == 'test':
        raise ValueError('once samples have been added to "test", they cannot be removed')

    cache_dir = utils.get_cache_dir()
    dataset_dir = os.path.join(cache_dir, p.dataset)

    # get the param set from the param number
    try:
        param_set = list(sorted(par for par in os.listdir(dataset_dir) if not par.startswith('.')))[p.param_num - 1]
    except FileNotFoundError:
        raise ValueError('dataset "{}" does not exist'.format(p.dataset))
    except IndexError:
        raise ValueError('param set number {} does not exist'.format(p.param_num))

    from_dir = os.path.join(dataset_dir, param_set, p.from_set.lower())
    to_dir = os.path.join(dataset_dir, param_set, p.to_set.lower())
    from_dir_nice = os.path.join(*from_dir[len(cache_dir):].split(os.path.sep)[3:])
    to_dir_nice = os.path.join(*to_dir[len(cache_dir):].split(os.path.sep)[3:])

    # get the counts for the sets under this param set
    if p.from_set.lower() == 'raw':
        from_count = utils.get_count(from_dir, p.verbose)
    else:
        from_count = utils.get_count(from_dir, p.verbose)['none']
    if p.to_set.lower() == 'raw':
        to_count = utils.get_count(to_dir, p.verbose)
    else:
        to_count = utils.get_count(to_dir, p.verbose)['none']

    if from_count < p.n_samples:
        raise ValueError('{} only has {} samples'.format(from_dir_nice, from_count))

    # ask confirmation
    print('Dataset:', p.dataset)
    print('Params:', param_set)
    print('Move {} samples from {} to {}?'.format(p.n_samples, from_dir_nice, to_dir_nice))
    print(
        'This would result in sizes {} and {}, respectively.'.format(from_count - p.n_samples, to_count + p.n_samples),
        end=''
    )
    confirm = input(' (y/N): ')
    if confirm.lower() not in {'y', 'yes'}:
        print('Aborting')
        return

    # perform move
    from_X, from_y = utils.from_pickle(os.path.join(from_dir, 'none'), p.verbose)
    to_X, to_y = utils.from_pickle(os.path.join(to_dir, 'none'), p.verbose)
    to_X.extend(from_X[:p.n_samples])
    to_y.extend(from_y[:p.n_samples])
    utils.to_pickle((to_X, to_y), os.path.join(to_dir, 'none'))
    utils.to_pickle((from_X[p.n_samples:], from_y[p.n_samples:]), os.path.join(from_dir, 'none'))

    # remove preprocessed versions of those samples from the source location
    if p.from_set.lower() != 'raw':
        for preproc_set in os.listdir(from_dir):
            if preproc_set != 'none':
                preproc_path = os.path.join(from_dir, preproc_set)
                from_X, from_y = utils.from_pickle(preproc_path, p.verbose)
                utils.to_pickle(
                    (from_X[min(p.n_samples, len(from_X)):], from_y[min(p.n_samples, len(from_y)):]), preproc_path)

    # update counts
    if p.from_set.lower() == 'raw':
        utils.update_counts(os.path.join(from_dir, 'none'), )


    # update counts



def main(p):
    """Use the parsed options to perform actions.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.command == 'list':
        _list(p)
    elif p.command == 'move':
        _move(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command line interface for SLURM_gen.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # required arguments
    parser.add_argument('command', help='command to perform. One of {"list", "move"}.')
    parser.add_argument('dataset', nargs='?', help='Dataset to perform the list or move command on.')
    parser.add_argument('-p', '--param_num', type=int, help='Param set number given by the list command.')
    parser.add_argument('-f', '--from_set', help='Subset of the dataset from which to move samples.')
    parser.add_argument('-t', '--to_set', help='Subset of the dataset to which to move samples.')
    parser.add_argument('-n', '--n_samples', type=int, help='Number of samples to move.')
    parser.add_argument('-v', '--verbose', action='store_true', help='print debug info while running the command.')

    args = parser.parse_args()

    if args.command == 'list':
        if args.param_num is not None:
            raise ValueError('--param_num must not be set for the "list" command')
        if args.from_set is not None:
            raise ValueError('--from_set must not be set for the "list" command')
        if args.to_set is not None:
            raise ValueError('--to_set must not be set for the "list" command')
        if args.n_samples is not None:
            raise ValueError('--n_samples must not be set for the "list" command')
    elif args.command == 'move':
        if args.param_num is None:
            raise ValueError('--param_num must be set for the "move" command')
        if args.from_set is None:
            args.from_set = 'raw'  # default
        if args.to_set is None:
            raise ValueError('--to_set must be set for the "move" command')
        if args.n_samples is None:
            raise ValueError('--n_samples must be set for the "move" command')

    main(args)
