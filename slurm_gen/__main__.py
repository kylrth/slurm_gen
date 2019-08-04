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
    if p.first_arg is None:
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
            single_list(p.first_arg.lower(), p.verbose)
        except FileNotFoundError:
            print('No dataset found by name {}'.format(p.first_arg.lower()))


def _move(p):
    """Move unlabeled samples into a particular set.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    # TODO: don't allow moving to 'train+val', but allow everything else
    # TODO: don't move out of 'test'
    # TODO: update counts


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
    parser.add_argument(
        'first_arg', nargs='?',
        help='Argument to the command. If the command is "list", this is the dataset to list counts for.')
    parser.add_argument('-v', '--verbose', action='store_true', help='print debug info while running the command.')

    main(parser.parse_args())
