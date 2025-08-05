# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="cy_sim_utils",
    ext_modules=cythonize("cy_sim_utils.pyx"),
    include_dirs=[numpy.get_include()],
)
