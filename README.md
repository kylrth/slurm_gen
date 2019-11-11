# SLURM_gen

This package automates the process of generating large amounts of data, providing a clean interface between your simulation and the SLURM workload manager. It also manages the datasets you choose to generate, and allows easy access to cached simulations that load quickly. If you need more data than you have, SLURM_gen lets you know how many more samples need to be generated. Soon it will also be able to tell you how much compute time is necessary to create those samples.

## Installation

```bash
pip install -e .  # don't forget the period
```

## Usage

SLURM_gen provides a simple command line interface to

- generate data samples,
- assign those samples to a particular dataset name, like 'train' or 'test', and
- track the number of samples generated for various datasets and parameters.

You can define your own datasets by simply writing a function that outputs feature-label pairs, and placing that function in the `datasets` module.

### Example

Start by using the `generator` decorator to define a new function in `datasets.py`.

```python
# (in datasets.py)
import numpy as np

#
# ... (other code already present in the file)
#

@generator(cache_every=500)  # specify the number of samples to generate between each save
def noisy_sine(size, params):  # every generator needs to take these two parameters
    """Generate noisy data from a sine wave.

    Entries in params:
    - 'left': the lower bound of the interval.
    - 'right': the upper bound of the interval.
    - 'scale': standard deviation of additive noise.

    Args:
        size (int): number of samples to produce.
        params: parameters to the experiment.
    Yields:
        (float): an x value sampled from a uniform distribution across the interval.
        (float): sin(x) + noise.
    """
    x = np.random.random(params.left, params.right, size)
    y = np.sin(x) + np.random.normal(loc=0, scale=params.scale, size)
    for stuff in zip(x, y):
        yield stuff
```

The `@generator` decorator converts your function into a dataset generator which can be used by the `generate` module to create cache files containing arbitrary numbers of samples. In the future, you will be able to achieve the same result by importing the decorator into your own Python module and using it to define functions there.

Now you can generate a thousand samples for that dataset like this:

```bash
python -m slurm_gen.generate wavelength_sweep -n 1000 -njobs 40 --mem_per_cpu 1GB --params "{'left': -1, 'right': 1, 'scale': 0.5}" --time "30" --verbose
```

In the example above, we submitted 40 SLURM jobs, splitting the 1000 samples evenly among them. We set parameters of the dataset to custom values, and each job received thirty minutes of run time.

If we don't provide `--time`, a value three standard deviations above the mean of previous runs is used, adapted to the number of samples per job.

```python
import slurm_gen

# generate a thousand samples
slurm_gen.data_gen.generate_data(
    dataset='wavelength_sweep',  # specify the dataset you want by naming a function defined in datasets.py
    size=1000,  # specify the number of samples
    params={'sidewall_angle': 88},  # change parameters to the generating function
    options={'time': '2:00:00'},  # specify SLURM options to be different from the default
    njobs=40,  # specify the number of SLURM jobs to use to create the dataset
    verbose=True  # show the SLURM command submitted
)

# load those 1000 samples as a training set
X, y = slurm_gen.data_loading.get_data(
    dataset='wavelength_sweep',
    subset='train',
    size=1000,
    params={'sidewall_angle': 88},
    preproc=square,  # apply arbitrary preprocessing functions
    batch_preproc=False  # vectorized preprocessors can be applied if batch_preproc is set to True
)
```

## TODO

- When lots of raw samples are being assigned, `pickle` causes really big memory usage. Split pickling into smaller units.
- Add example of parameter object for sine.
- Allow datasets to be defined outside datasets.py.
- Let the generator decorator get information from the docstring.
