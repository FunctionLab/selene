import os

from Cython.Build import cythonize
import numpy as np
from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension

from selene import __version__


with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as readme:
    long_description = readme.read()

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
      description=("framework for developing sequence-level "
                   "deep learning networks"),
      long_description=long_description,
      long_description_context_type="text/markdown",
      packages=find_packages(),
      url="https://github.com/FunctionLab/selene",
      package_data={
        "selene": ["interpret/data/gencode_v28_hg38/*", "interpret/data/gencode_v28_hg19/*"]
      },
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
      ),
      ext_modules=cythonize(ext_modules),
      install_requires=[
        "h5py>=2.7.1",
        "matplotlib==2.2.2",
        "numpy>=1.13.3",
        "pandas>=0.20.3",
        "pyfaidx==0.5.3.1",
        "pytabix==0.0.2",
        "pytorch>=0.2.0",
        "scikit-learn>=0.19.0",
        "scipy>=0.19.1",
        "seaborn>=0.8.1",
        "statsmodels>=0.9.0",
        "yaml>=0.1.7"
      ])
