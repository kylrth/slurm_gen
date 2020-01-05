import numpy as np

import slurm_gen


class NoisySineParams(slurm_gen.DefaultParamObject):
    """Attributes defining parameters to the noisy_sine experiment."""

    # leftmost allowed value for x
    left = -1

    # rightmost allowed value for x
    right = 1

    # standard deviation of noise to add to sin(x)
    std_dev = 0.1


# we can specify extra SLURM batch parameters here
options = "--qos=test"


# here we also tell SLURM_gen to request 1GB of memory and save every 50 samples
@slurm_gen.dataset(NoisySineParams, "1GB", 50, options)
def noisy_sine(size, params):
    """Create samples from a noisy sine wave.

    Args:
        size (int): number of samples to generate.
        params (NoisySineParams): parameters to the experiment.
    Yields:
        (float): x-value.
        (float): y-value plus noise.
    """
    for x in np.random.uniform(params.left, params.right, size=size):
        yield x, np.sin(x) + np.random.normal(scale=params.std_dev)


@noisy_sine.preprocessor
def square_both(X, y):
    """Square both the inputs and the outputs."""
    return [ex ** 2 for ex in X], [wai ** 2 for wai in y]
