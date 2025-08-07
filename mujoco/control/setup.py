# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Lista de extensiones Cython que se compilarán
extensions = [
    Extension(
        name="cython_modules.cy_sim_utils",
        sources=["cython_modules/cy_sim_utils.pyx"],
        include_dirs=[np.get_include()],
        language="c++",  # quítalo si no usas C++
    ),
    Extension(
        name="cython_modules.fast_mujoco_utils",
        sources=["cython_modules/fast_mujoco_utils.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    ),
]

setup(
    name="cython_modules",
    version="0.1.0",
    description="Funciones optimizadas en Cython para simulación",
    author="Tu Nombre",
    packages=["cython_modules"],  # asegura que la carpeta se instale
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
