from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

genome_module = Extension(
    "selene.sequences._genome",
    ["selene/sequences/_genome.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "selene.targets._genomic_features",
    ["selene/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]

setup(name="selene",
      version="0.0.0",
      description="framework for training sequence-level " \
                  "deep learning networks",
      packages=["selene"],
      ext_modules=cythonize(ext_modules))
