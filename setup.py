from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize
from selene import __version__


genome_module = Extension(
    "selene.sequences._sequence",
    ["selene/sequences/_sequence.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "selene.targets._genomic_features",
    ["selene/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]

setup(name="selene",
      version=__version__,
      description=("framework for training sequence-level "
                   "deep learning networks"),
      packages=["selene"],
      ext_modules=cythonize(ext_modules))
