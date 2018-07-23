from setuptools import find_packages
from setuptools import setup

from selene_sdk import __version__


try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.extension import Extension
    USING_CYTHON = False
else:
    USING_CYTHON = True

ext = '.pyx' if USING_CYTHON else '.c'

genome_module = Extension(
    "selene_sdk.sequences._sequence",
    ["selene_sdk/sequences/_sequence" + ext])
    #include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "selene_sdk.targets._genomic_features",
    ["selene_sdk/targets/_genomic_features" + ext])
    #include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]
cmdclass = {'build_ext': build_ext} if USING_CYTHON else {}

setup(name="selene-sdk",
      version=__version__,
      description=("framework for developing sequence-level "
                   "deep learning networks"),
      packages=find_packages(),
      url="https://github.com/FunctionLab/selene",
      package_data={
        "selene-sdk": ["interpret/data/gencode_v28_hg38/*", "interpret/data/gencode_v28_hg19/*"]
      },
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
      ),
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      install_requires=[
        "h5py>=2.7.1",
        "matplotlib==2.2.2",
        "numpy>=1.13.3",
        "pandas>=0.20.3",
        "pyfaidx==0.5.3.1",
        "pytabix==0.0.2",
        "torch>=0.1.2",
        "torchvision>=0.1.9",
        "scikit-learn>=0.19.0",
        "scipy>=0.19.1",
        "seaborn>=0.8.1",
        "statsmodels>=0.9.0",
        "pyyaml>=3.12"
      ])
