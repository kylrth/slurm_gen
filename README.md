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

You can define your own datasets simply by writing a function that outputs feature-label pairs. Define that function in a file called `datasets.py`, and point SLURM_gen at the directory containing that file.

### Example

Here we'll show how to define a simple dataset, generate some samples, and access them.

#### Define the generator

Start by using the `DefaultParamObject` class and the `@dataset` decorator to define a new dataset. These definitions should be placed in a Python file called `datasets.py`.

```python
# example/datasets.py
import numpy as np

from slurm_gen import DefaultParamObject, dataset


class NoisySineParams(DefaultParamObject):
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
@dataset(NoisySineParams, "1GB", 50, options)
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

The `NoisySineParams` defines the possible configuration parameters that the generator can accept, as well as the default values for those parameters. When generating or accessing samples, we can specify non-default values for any of these parameters.

The `@dataset` decorator converts `noisy_sine` into a dataset which can be used by the `slurm_gen.generate` module to create cache files containing arbitrary numbers of samples. We can define as many functions as we like in `datasets.py`, and all those marked with `@dataset` will be usable in SLURM_gen.

#### Generate samples

Now that we've defined the generator, we can generate some samples for that dataset like this:

```bash
cd example/  # the directory containing datasets.py
python -m slurm_gen.generate noisy_sine -n 1000 --njobs 40 --time "30"
python -m slurm_gen.generate noisy_sine -n 1000 --njobs 40 --params "{'left': 0, 'std_dev': 0.5}"
```

In the first example above, we submitted 40 SLURM jobs, splitting the 1000 samples evenly among them. Since we had no samples for this dataset yet, we had to provide `--time`. In the second example, we omitted the `--time` argument, and a time duration three standard deviations above the mean of previous runs was used, adapted to the number of samples per job. In the second example we also set some configuration parameters to non-default values.

#### Managing samples

We can list the available samples from the command line:

```bash
cd example/
python -m slurm_gen.list
```

The output will look like this:

```txt
noisy_sine:
Param set #0:
    left#-1|right#1| raw: 1000
        std_dev#0.1|
Param set #1:
    left#0|right#1| raw: 1000
       std_dev#0.5|
```

We can see the samples for the "noisy_sine" dataset divided into sets by the parameters given.

If we want to move some of those samples into a group labeled "train", we can do so like this:

```bash
cd example/
python -m slurm_gen.move noisy_sine 700 train -p 0
```

The `-p` argument identifies which parameter set to use. You can also use a dictionary of values as the identifier, by passing a string that will be evaluated as a dictionary.

After the move, the output of `python -m slurm_gen.list` will be

```txt
noisy_sine:
Param set #0:
    left#-1|right#1| raw: 300
        std_dev#0.1| train: unprocessed(700)
Param set #1:
    left#0|right#1| raw: 1000
       std_dev#0.5|
```

Once you've moved samples into a labeled group, you can't move them back. This is to avoid accidentally mixing samples between groups, possibly inflating the accuracy of machine learning models.

#### Preprocessing samples

You may have noticed that `slurm_gen.list` noted 700 "unprocessed" samples. Once samples are in a group, you can apply preprocessors to them. Preprocessors must be defined in the same `datasets.py` file. To continue the example, add the following preprocessor for our `noisy_sine` dataset.

```python
# added to datasets.py
@noisy_sine.preprocessor
def square_both(X, y):
    """Square both the inputs and the outputs."""
    return [ex ** 2 for ex in X], [wai ** 2 for wai in y]
```

Note that the preprocessor is defined for one particular dataset. If the same preprocessor needs to be defined for multiple datasets, just add the decorators one after the other.

Preprocess some samples from 'train' by running the following command:

```bash
python -m slurm_gen.preprocess noisy_sine square_both train 600 -p 0
```

After the data is preprocessed, the output of `python -m slurm_gen.list` will be

```txt
noisy_sine:
Param set #0:
    left#-1|right#1| raw: 300
                   | train: unprocessed(700)
                   |      : square_both(600)
Param set #1:
    left#0|right#1| raw: 1000
       std_dev#0.5|
```

#### Accessing the samples

To access the samples within Python, use the `get_dataset` function:

```python
from slurm_gen import Cache

# load those 700 samples as a training set
X, y = Cache("./example/")["noisy_sine"][0]["train"].get(700)
```

## TODO

- Define the object hierarchy in the readme.
- Get rid of utils.py?
- Be more efficient with keeping track of the sizes of the datasets.
- Be able to preprocess on a SLURM job.
