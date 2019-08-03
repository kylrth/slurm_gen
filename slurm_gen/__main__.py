"""Command line interface for SLURM_gen.

Kyle Roth. 2019-07-27.
"""


import argparse
import os

from slurm_gen import utils


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

    name_col_len = max(len(d) for dataset in counts for d in counts[dataset])
    raw_col_len = max(len(str(counts[dataset][d]['raw'])) for dataset in counts for d in counts[dataset])
    train_col_len = max(len(str(counts[dataset][d]['train'])) for dataset in counts for d in counts[dataset])
    val_col_len = max(len(str(counts[dataset][d]['val'])) for dataset in counts for d in counts[dataset])
    test_col_len = max(len(str(counts[dataset][d]['test'])) for dataset in counts for d in counts[dataset])

    print('Name'.ljust(name_col_len + 4), 'Raw'.ljust(raw_col_len), 'Train'.ljust(train_col_len), sep='|', end='')
    print('Val'.ljust(val_col_len), 'Test'.ljust(test_col_len), sep='|')

    print(counts)


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
