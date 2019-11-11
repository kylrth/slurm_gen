"""Makes it easy to generate and handle arbitrarily-sized datasets on a SLURM HPC
environment.

Kyle Roth. 2019-06-05.
"""


from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SLURM_gen",
    version="0.1a3",
    description=__doc__.split("\n")[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url
    author="Kyle Roth",
    author_email="kylrth@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Software Development :: User Interfaces",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="SLURM machine-learning data",
    py_modules=["generate", "data_loading", "datasets", "utils"],
    install_requires=["numpy", "matplotlib", "scipy", "pyMode"],
    # we need to be able to write the cache to the location where the module lives
    zip_safe=False,
)
