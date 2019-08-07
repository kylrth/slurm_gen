"""Command line interface for SLURM_gen.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen import utils
from slurm_gen.data_objects import Cache


def print_sizes(param_set):
    """Print (nicely) the ParamSet and its sizes.

    Parameter strings are often very long, so we print them (and their counts) nicely.

    Args:
        param_set (ParamSet)
    """
    # collect strings of parameters two wide
    param_iter = iter(param_set.name.split('|'))
    param_strings = []
    while True:
        try:
            param_strings.append(next(param_iter) + '|')
            param_strings[-1] += next(param_iter) + '|'
        except StopIteration:
            break

    # collect strings of counts
    count_strings = [' raw: {}'.format(param_set.raw_size)]
    for group in param_set:
        count_strings.append(' {}: unprocessed({})'.format(group.name, group.unprocessed_size))
        for preproc_set in group:
            count_strings.append(' ' * (len(group.name) + 1) + ': {}({})'.format(preproc_set.name, len(preproc_set)))

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


def single_list(dataset):
    """Print the count information for a single dataset.

    Args:
        dataset (Dataset)
    """
    count = 0
    for param_set in dataset:
        print('Param set #{}:'.format(count))
        print_sizes(param_set)
        count += 1


def _list(p):
    """List the datasets along with the number of samples generated for them.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.dataset is None:
        # list all of them!
        divider = '-' * 80 + '\n'
        did_print = False

        for dataset in Cache():
            if did_print:
                print(divider)
            print('{}:'.format(dataset.name))
            single_list(dataset)
            did_print = True

        if not did_print:
            print('No datasets found. Generate one!')
    else:
        # get list the one dataset
        single_list(Cache()[p.dataset])


def _move(p):
    """Move samples between sets, confirming the choice before doing so.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    if p.to_set == 'train+val':
        raise ValueError('the name "train+val" is reserved for combining the "train" and "val" sets')
    if p.from_set == 'test':
        raise ValueError('once samples have been added to "test", they cannot be removed')

    cache = Cache()
    dataset = cache[p.dataset]
    param_set = dataset[p.param_num]

    from_group = param_set[p.from_set]
    to_group = param_set[p.to_set]

    # currently working here.............................................
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
