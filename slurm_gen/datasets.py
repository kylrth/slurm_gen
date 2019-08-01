"""Functions defining how each dataset is generated, to be called by the data generation functions defined in
data_gen.py.

Each function should have call signature (size, params), where size is the number of examples to produce and params is a
dict of parameters to accept.

Kyle Roth. 2019-05-17.
"""


from functools import wraps
import os
import pickle
from time import time

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import pyMode as pm
from pyMode.materials import Si, SiO2

from slurm_gen import utils


def generator(cache_every, ParamClass):
    """Create a decorator that turns a data generating function into one that stores data in the raw cache, caching
    every cache_every data points.

    Args:
        cache_every (int): number of data points to load between caching.
        ParamClass (class): class name of the parameter object that the function accepts. Must have a to_string() method
                            that creates a string of the parameter values, and a constructor that creates a new object
                            from a dict of params that override defaults. It suffices to be a subclass of
                            utils.DefaultParamObject.
    Returns:
        (decorator): decorator for a generating function that creates the caching function as described.
    """
    def decorator(f):
        """Turn a data generating function into one that stores data in the raw cache, caching every cache_every data
        points.

        Args:
            f (function): data generating function with call signature (size, params).
        Returns:
            (function): data generating function that stores the output of the original function in the cache.
        """
        fn_name = utils.get_func_name(f)

        @wraps(f)
        def wrapper(size, params):
            """Write the results of the function to {cache_dir}/{dataset}/{params}/raw/{somefile}.pkl.

            Args:
                size (int): number of samples to create.
                params (dict): parameters to pass to generating function.
            """
            # convert to the parameter class
            params = ParamClass(**params)

            # cache it in the raw location
            dataset_dir = utils.get_dataset_dir(fn_name, params)
            raw_path = os.path.join(
                dataset_dir,
                'raw/{}_{{}}.pkl'.format(utils.get_unique_filename())
            )
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)

            iter_data = f(size, params)

            data_count = 0
            to_store_x = []
            to_store_y = []

            # create a file to store times
            time_file = os.path.join(dataset_dir, '.times/{}.time').format(utils.get_unique_filename())
            os.makedirs(os.path.dirname(time_file), exist_ok=True)

            with open(time_file, 'a+') as time_f:
                while True:
                    start = time()

                    # try to get the next pair
                    try:
                        x, y = next(iter_data)
                    except StopIteration:
                        # store everything left over
                        with open(raw_path.format(data_count // cache_every + 1), 'wb') as outfile:
                            pickle.dump((to_store_x, to_store_y), outfile)
                        break

                    # add it to the temporary list
                    to_store_x.append(x)
                    to_store_y.append(y)
                    data_count += 1

                    # store it every `cache_every` iterations
                    if not data_count % cache_every:
                        with open(raw_path.format(data_count // cache_every), 'wb') as outfile:
                            pickle.dump((to_store_x, to_store_y), outfile)

                    # record the time spent
                    time_f.write(str(time() - start) + '\n')

        # store the class used by this function, for use in data_loading.get_data
        wrapper.paramClass = ParamClass

        return wrapper

    return decorator


def make_grid(xs, hs):
    """Creates a 2D grid with the density specified by xs and hs.

    Used by pyMode for simulation points.

    Args:
        xs (array-like): positions at which to switch densities.
        hs (array-like): densities to use between positions.
    Returns:
        (np.ndarray): linspace of points at the appropriate density for each section.
    """
    grid = [min(xs)]
    interp = interp1d(xs, hs, kind='linear')
    while grid[-1] < max(xs):
        h = interp(grid[-1])
        grid.append(grid[-1] + h)
    return np.array(grid)


def _single_sim(params):
    """Perform a single simulation using the specified parameters.

    Necessary attributes of params:
        sidewall_angle (Number): angle of waveguide wall.
        width (Number): width of top of waveguide.
        thickness (Number): height of waveguide.
        xWidth (Number): width of simulation in microns.
        yWidth (Number): height of simulation in microns.
        dcore (Number): density of simulation in waveguide core.
        dcladding (Number): density of simulation in waveguide cladding.
        wavelength (Number): wavelength of light in micrometers.
        numModes (int): number of modes to solve for.
    Returns:
        (list): wave numbers of modes solved for.
        (list): radius (r) of H field for each mode.
        (list): height (z) of H field for each mode.
        (list): angle (phi) of H field for each mode.
        (list): radius (r) of E field for each mode.
        (list): height (z) of E field for each mode.
        (list): angle (phi) of E field for each mode.
    """
    # set up the waveguide geometry
    sidewall_angle_radians = params.sidewall_angle / 180 * np.pi
    bottomFace = params.width + 2 * (params.thickness / np.tan(sidewall_angle_radians))

    # create shape
    waveguide = pm.Trapezoid(
        center=pm.Vector3(0, 0),
        top_face=params.width,
        thickness=params.thickness,
        sidewall_angle=sidewall_angle_radians,
        core=Si,
        cladding=SiO2
    )

    geometry = [waveguide]

    # Set up the simulation grid
    xLocs = [0, bottomFace / 2, bottomFace / 2 + 0.1, params.xWidth / 2]
    xVals = [params.dcore, params.dcore, params.dcladding, params.dcladding]
    xx = make_grid(xLocs, xVals)

    yLocs = [0, params.thickness / 2, params.thickness / 2 + 0.1, params.yWidth / 2]
    yVals = [params.dcore, params.dcore, params.dcladding, params.dcladding]
    yy = make_grid(yLocs, yVals)
    yy = np.concatenate((-(np.flip(yy[1:-1], 0)), yy))

    # Run the simulation
    sim = pm.Simulation(
        geometry=geometry,
        wavelength=params.wavelength,
        numModes=params.numModes,
        xGrid=xx,
        yGrid=yy,
        background=SiO2,
        # don't overlap data files for this simulation with concurrent simulations
        folderName=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cache/.temp_wgms3d',
            utils.get_unique_filename()
        )
    )

    sim.run()
    return sim.getFields()


def test__single_sim(**kwargs):
    """Run a simple simulation with default values using single_sim.

    kwargs will replace default parameters passed to single_sim.
    """
    # set parameters of the experiment
    p = WavelengthSweepParams(**kwargs)  # overwrite defaults using parameters passed in.

    for i, thing in enumerate(_single_sim(p)[1:]):
        # plot each image and save it to kyle/taper_modeling/figures
        plt.imshow(thing[0].astype(np.float), cmap='gray', alpha=0.5)
        os.makedirs('kyle/taper_modeling/figures/{}'.format(i), exist_ok=True)
        plt.savefig('kyle/taper_modeling/figures/{}/straight_{}.png'.format(i, '_'.join(str(thing)
                                                                                        for thing in kwargs.values())))
        plt.clf()


class WavelengthSweepParams(utils.DefaultParamObject):
    """Attributes defining parameters to the wavelength_sweep experiment."""
    # angle of incidence between waveguide side wall and substrate
    sidewall_angle = 90

    # dimensions of waveguide in microns
    width = 0.5
    thickness = 0.22

    # dimensions of simulation in microns
    xWidth = 5
    yWidth = 3

    # simulation resolution in the waveguide core
    dcore = 20.01e-4

    # simulation resolution in the cladding
    dcladding = 20.01e-3

    # number of modes to return
    numModes = 1

    # choice of minimum and maximum wavelength for sweep
    wl_left = 1.45
    wl_right = 1.65


@generator(2, WavelengthSweepParams)
def wavelength_sweep(size, params):
    """Return all six field profiles for a straight and square waveguide, varying wavelength and nothing else.

    Args:
        size (int): number of samples to create.
        params (WavelengthSweepParams): parameters to the experiment.
    Yields:
        (float): random wavelength.
        (np.ndarray): profile images.
    """
    for wl in np.random.uniform(params.wl_left, params.wl_right, size=size):  # pylint:disable=no-member
        params.wavelength = wl
        fields = _single_sim(params)
        yield wl, fields


class DimensionSweepParams(utils.DefaultParamObject):
    """Attributes defining parameters to the dimension_sweep experiment."""
    # angle of incidence between waveguide side wall and substrate
    sidewall_angle = 90

    # dimensions of waveguide in microns
    width_left = 0.2
    width_right = 0.7
    thickness_left = 0.15
    thickness_right = 0.5

    # dimensions of simulation in microns
    xWidth = 5
    yWidth = 3

    # simulation resolution in the waveguide core
    dcore = 20.01e-4

    # simulation resolution in the cladding
    dcladding = 20.01e-3

    # number of modes to return
    numModes = 1

    # wavelength
    wavelength = 1.55


@generator(2, DimensionSweepParams)
def dimension_sweep(size, params):
    """Return all six field profiles for a straight and square waveguide, varying width and thickness.

    Args:
        size (int): number of samples to create.
        params (DimensionSweepParams): parameters to the experiment.
    Yields:
        (tuple(float, float)): random width and height.
        (np.ndarray): profile images.
    """
    for width, thickness in zip(
            np.random.uniform(params.width_left, params.width_right, size=size),  # pylint:disable=no-member
            np.random.uniform(params.thickness_left, params.thickness_right, size=size)  # pylint:disable=no-member
    ):
        params.width = width
        params.thickness = thickness
        fields = _single_sim(params)
        yield (width, thickness), fields
