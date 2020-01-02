# SLURM_gen

This package automates the process of generating large amounts of data, providing a clean interface between your simulation and the SLURM workload manager. It also manages the datasets you choose to generate, and allows easy access to cached simulations that load quickly. If you need more data than you have, SLURM_gen lets you know how many more samples need to be generated, and how much compute time it will take.

## Installation

```bash
pip install -e .  # don't forget the period
```

## Usage

SLURM_gen provides a simple command line interface to

- generate data samples,
- assign those samples to a particular dataset name, like 'train' or 'test', and
- track the number of samples generated for various datasets and parameters.

You can define your own datasets by simply writing a function that outputs feature-label pairs.

### Example

Here we'll show how to define a simple dataset, generate some samples, and access them.

#### Define the generator

Start by using the `DefaultParamObject` class and `generator` decorator to define a new function in a dedicated Python file.

```python
# noisy_sine_example.py
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


@slurm_gen.generator(50, NoisySineParams)
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

```

The `slurm_gen.DefaultParamObject` defines the possible configuration parameters that the generator can accept, as well as the default values for those parameters. When generating or accessing samples, we can specify non-default values for any of these parameters.

The `@slurm_gen.generator` decorator converts `noisy_sine` into a dataset generator which can be used by the `generate` module to create cache files containing arbitrary numbers of samples. If multiple functions in the file are decorated with `@slurm_gen.generator`, only the first one is used.

#### Generate samples

Now that we've defined the generator, we can generate a thousand samples for that dataset like this:

```bash
python -m slurm_gen.generate ./noisy_sine_example.py -n 1000 -njobs 40 --mem_per_cpu 1GB --params "{'left': 0, 'scale': 0.5}" --time "30" --verbose
```

In the example above, we submitted 40 SLURM jobs, splitting the 1000 samples evenly among them. We set some configuration parameters to non-default values, and each job received thirty minutes of run time.

If we don't provide `--time`, a value three standard deviations above the mean of previous runs is used, adapted to the number of samples per job. If this is the first time this dataset is being generated with this set of config parameters, `--time` must be specified.

#### Accessing the samples

We can list the available samples from the command line:

```bash
python -m slurm_gen.list ./noisy_sine_example.py
```

The output will look like this:

```txt
[TODO]
```

To access the samples within Python, use the `get_dataset` function:

```python
from slurm_gen import get_dataset

# load those 1000 samples as a training set
X, y = get_dataset("./noisy_sine_example.py")[0]["train"].get(1000)
```

## TODO

- Be more efficient with keeping track of the sizes of the datasets.
