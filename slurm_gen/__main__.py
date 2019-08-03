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
    param_width = max(len(p_str) for p_str in param_strings)

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


def _list(p):
    """List the datasets along with the number of samples generated for them.

    Args:
        p (argparse.Namespace): namespace object containing the attributes specified below in the parser definition.
    """
    counts = {}
    for dataset in os.listdir(utils.get_cache_dir()):
        if not dataset.startswith('.'):
            utils.v_print(p.verbose, 'Getting counts for dataset "{}"'.format(dataset))
            counts[dataset] = utils.get_counts(dataset, p.verbose)

    divider = '-' * 80

    for dataset in counts:
        print(divider)
        print('{}:'.format(dataset.upper()))
        count = 1
        for params in counts[dataset]:
            print('Param set #{}:'.format(count))
            print_sizes(params, counts[dataset][params])
            count += 1
    print(divider)


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
    parser.add_argument('command', help='command to perform. One of {"list"}')
    parser.add_argument('-v', '--verbose', action='store_true', help='print debug info while running the command.')

    main(parser.parse_args())
